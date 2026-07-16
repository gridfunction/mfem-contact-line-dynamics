/**
 * @file ch2d_conservative_wetting.cpp
 * @brief Fully implicit Cahn--Hilliard wetting/dewetting benchmark.
 *
 * This code implements the mixed finite element scheme
 *
 *   (xi (phi^n-phi^{n-1})/dt, eta) + (grad mu^n, grad eta) = 0,
 *
 *   -(mu^n, psi) + sigma_LG delta (grad phi^n, grad psi)
 *   + (sigma_LG/delta F'(phi^n), psi)
 *   + < zeta delta (phi^n-phi^{n-1})/dt
 *       + 4/(3 sqrt(2)) (sigma_SL-sigma_SG) f_w'(phi^n), psi >_Gamma = 0.
 *
 * The Newton system is assembled as a 2-by-2 block operator in the unknowns
 * (delta phi, delta mu) and solved by GMRES.  The preconditioner is a simple
 * block diagonal approximation: an AMG solve for a positive elliptic
 * approximation of the phi block, and a signed Jacobi mass solve for the mu
 * block.  This keeps the first implementation faithful to the manuscript
 * equations without assembling a dense H^{-1} Schur complement.
 */

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "mfem.hpp"  // NOLINT [no include directory when naming mfem include file]

namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;

struct Options {
  int nx = 128;
  int ny = 128;
  int order = 1;
  double x_min = 0.0;
  double x_max = 1.0;
  double y_min = 0.0;
  double y_max = 1.0;
  double droplet_x_min = 0.0;
  double droplet_x_max = 0.5;
  double droplet_y_min = 0.0;
  double droplet_y_max = 0.25;
  double theta_deg = 45.0;
  double delta = 0.01;
  double dt = 0.01;
  double min_dt = 1.0e-4;
  double t_final = 4.0;
  double energy_decay_rate_tol = 1.0e-7;
  double sigma_lg = 1.0;
  double xi = 1.0;
  double zeta = 1.0;
  double newton_tol = 1.0e-10;
  int newton_maxit = 20;
  double linear_tol = 1.0e-6;
  int linear_maxit = 2000;
  int preconditioner_rebuild_iteration_limit = 30;
  int max_dt_retries = 8;
  int output_every = 0;
  std::string preconditioner = "direct";
  std::string output_dir;
  bool visualization = true;
  bool adaptive_time_step = true;
};

double DoubleWell(double phi) { return 0.25 * (phi * phi - 1.0) * (phi * phi - 1.0); }

double DoubleWellPrime(double phi) { return (phi * phi - 1.0) * phi; }

double DoubleWellSecond(double phi) { return 3.0 * phi * phi - 1.0; }

double WallEnergy(double phi) {
  if (phi > 1.0) {
    return 0.75 * (phi - 1.0) * (phi - 1.0);
  }
  if (phi >= -1.0) {
    return 0.25 * (phi * phi * phi - 3.0 * phi) + 0.5;
  }
  return 1.0 - 0.75 * (phi + 1.0) * (phi + 1.0);
}

double WallEnergyPrime(double phi) {
  if (phi > 1.0) {
    return 1.5 * (phi - 1.0);
  }
  if (phi >= -1.0) {
    return 0.75 * (phi * phi - 1.0);
  }
  return -1.5 * (phi + 1.0);
}

double WallEnergySecond(double phi) {
  if (phi > 1.0) {
    return 1.5;
  }
  if (phi >= -1.0) {
    return 1.5 * phi;
  }
  return -1.5;
}

double SigmaSLMinusSG(const Options& opts) {
  return -opts.sigma_lg * std::cos(opts.theta_deg * kPi / 180.0);
}

std::string NumberToken(double value) {
  std::ostringstream os;
  os << std::fixed << std::setprecision(6) << value;
  std::string token = os.str();
  while (token.size() > 1 && token.back() == '0') {
    token.pop_back();
  }
  if (!token.empty() && token.back() == '.') {
    token.pop_back();
  }
  for (char& ch : token) {
    if (ch == '-') {
      ch = 'm';
    } else if (ch == '.') {
      ch = 'p';
    }
  }
  return token;
}

bool HasOutputDirectoryOption(int argc, char* argv[]) {
  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    if (arg == "-out" || arg == "--output-dir") {
      return true;
    }
  }
  return false;
}

bool HasFinalTimeOption(int argc, char* argv[]) {
  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    if (arg == "-tf" || arg == "--t-final") {
      return true;
    }
  }
  return false;
}

double DefaultFinalTime(const Options& opts) {
  if (std::abs(opts.theta_deg - 45.0) < 1.0e-8) {
    return 3.44;
  }
  if (std::abs(opts.theta_deg - 135.0) < 1.0e-8) {
    return 2.39;
  }
  return opts.t_final;
}

std::string DefaultOutputDirectory(const Options& opts) {
  std::ostringstream os;
  os << "Saves/ch2d_conservative_wetting"
     << "_theta" << NumberToken(opts.theta_deg)
     << "_nx" << opts.nx
     << "_ny" << opts.ny
     << "_delta" << NumberToken(opts.delta)
     << "_dt" << NumberToken(opts.dt);
  return os.str();
}

bool NextPinnedOutputTime(double time, double proposed_end, double& pinned_time) {
  constexpr double pinned_times[] = {0.02, 0.2, 1.0};
  constexpr double tol = 1.0e-12;
  for (const double target : pinned_times) {
    if (target > time + tol && target <= proposed_end + tol) {
      pinned_time = target;
      return true;
    }
  }
  return false;
}

bool IsPinnedOutputTime(double time) {
  constexpr double pinned_times[] = {0.02, 0.2, 1.0};
  constexpr double tol = 1.0e-10;
  for (const double target : pinned_times) {
    if (std::abs(time - target) <= tol) {
      return true;
    }
  }
  return false;
}

class RectangularDropletCoefficient : public mfem::Coefficient {
 public:
  explicit RectangularDropletCoefficient(const Options& opts) : opts_(opts) {}

  double Eval(mfem::ElementTransformation& trans,
              const mfem::IntegrationPoint& ip) override {
    mfem::Vector xx;
    trans.Transform(ip, xx);
    const double indicator =
        BoundaryAnchoredIndicator(xx(0), opts_.x_min, opts_.droplet_x_min,
                                  opts_.droplet_x_max,
                                  (opts_.x_max - opts_.x_min) / opts_.nx) *
        BoundaryAnchoredIndicator(xx(1), opts_.y_min, opts_.droplet_y_min,
                                  opts_.droplet_y_max,
                                  (opts_.y_max - opts_.y_min) / opts_.ny);
    return 1.0 - 2.0 * indicator;
  }

 private:
  static double BoundaryAnchoredIndicator(double x, double domain_min, double droplet_min,
                                          double droplet_max, double h) {
    const double eps = 1.0e-10 * std::max(h, 1.0);
    if (std::abs(droplet_min - domain_min) > eps) {
      return OneDimensionalIndicator(x, droplet_min, droplet_max);
    }

    const double grid_pos = (x - domain_min) / h;
    const double node_index_real = std::round(grid_pos);
    if (std::abs(grid_pos - node_index_real) > 1.0e-8) {
      return OneDimensionalIndicator(x, droplet_min, droplet_max);
    }

    const int node_index = static_cast<int>(node_index_real);
    const double cut_pos = (droplet_max - domain_min) / h;
    const int cut_cell = static_cast<int>(std::floor(cut_pos));
    const double fraction = cut_pos - cut_cell;
    if (fraction < 1.0e-12 || 1.0 - fraction < 1.0e-12) {
      const int interface_node = (fraction < 1.0e-12) ? cut_cell : cut_cell + 1;
      if (node_index < interface_node) {
        return 1.0;
      }
      if (node_index == interface_node) {
        return 0.5;
      }
      return 0.0;
    }
    if (fraction < 0.5) {
      if (node_index < cut_cell) {
        return 1.0;
      }
      if (node_index == cut_cell) {
        return cut_cell == 0 ? 2.0 * fraction : 0.5 + fraction;
      }
      return 0.0;
    }
    if (node_index <= cut_cell) {
      return 1.0;
    }
    if (node_index == cut_cell + 1) {
      return fraction - 0.5;
    }
    return 0.0;
  }

  static double OneDimensionalIndicator(double x, double xmin, double xmax) {
    const double eps = 1.0e-12;
    if (x < xmin - eps || x > xmax + eps) {
      return 0.0;
    }
    if (std::abs(x - xmax) <= eps) {
      return 0.5;
    }
    return 1.0;
  }

  const Options& opts_;
};

void ShiftMeshToDomain(mfem::Mesh& mesh, const Options& opts) {
  for (int i = 0; i < mesh.GetNV(); ++i) {
    double* vertex = mesh.GetVertex(i);
    vertex[0] += opts.x_min;
    vertex[1] += opts.y_min;
  }
}

void AddElementMatrix(mfem::SparseMatrix& mat, const mfem::Array<int>& dofs,
                      const mfem::DenseMatrix& elmat) {
  const int nd = dofs.Size();
  for (int i = 0; i < nd; ++i) {
    for (int j = 0; j < nd; ++j) {
      mat.Add(dofs[i], dofs[j], elmat(i, j));
    }
  }
}

void AssembleCHResidualAndJacobian(mfem::ParFiniteElementSpace& fes,
                                   const mfem::ParGridFunction& phi,
                                   const mfem::ParGridFunction& mu,
                                   const mfem::ParGridFunction& phi_old,
                                   const Options& opts, mfem::Vector& r_phi,
                                   mfem::Vector& r_mu, mfem::SparseMatrix& j00,
                                   mfem::SparseMatrix& j01, mfem::SparseMatrix& j10,
                                   mfem::SparseMatrix& j11) {
  const int ndofs = fes.GetVSize();
  r_phi.SetSize(ndofs);
  r_mu.SetSize(ndofs);
  r_phi = 0.0;
  r_mu = 0.0;

  const double wall_factor = (4.0 / (3.0 * std::sqrt(2.0))) * SigmaSLMinusSG(opts);
  mfem::Array<int> dofs;
  mfem::Vector el_phi, el_mu, el_phi_old, shape, grad_phi, grad_mu;
  mfem::DenseMatrix dshape, e00, e01, e10, e11;

  for (int e = 0; e < fes.GetNE(); ++e) {
    const mfem::FiniteElement* fe = fes.GetFE(e);
    mfem::ElementTransformation* trans = fes.GetElementTransformation(e);
    fes.GetElementDofs(e, dofs);
    phi.GetSubVector(dofs, el_phi);
    mu.GetSubVector(dofs, el_mu);
    phi_old.GetSubVector(dofs, el_phi_old);

    const int nd = fe->GetDof();
    const int dim = fe->GetDim();
    shape.SetSize(nd);
    dshape.SetSize(nd, dim);
    grad_phi.SetSize(dim);
    grad_mu.SetSize(dim);
    e00.SetSize(nd);
    e01.SetSize(nd);
    e10.SetSize(nd);
    e11.SetSize(nd);
    e00 = 0.0;
    e01 = 0.0;
    e10 = 0.0;
    e11 = 0.0;

    const mfem::IntegrationRule& ir =
        mfem::IntRules.Get(fe->GetGeomType(), 2 * fe->GetOrder() + trans->OrderW() + 2);
    for (int q = 0; q < ir.GetNPoints(); ++q) {
      const mfem::IntegrationPoint& ip = ir.IntPoint(q);
      trans->SetIntPoint(&ip);
      fe->CalcShape(ip, shape);
      fe->CalcPhysDShape(*trans, dshape);
      dshape.MultTranspose(el_phi, grad_phi);
      dshape.MultTranspose(el_mu, grad_mu);

      const double phi_q = shape * el_phi;
      const double mu_q = shape * el_mu;
      const double phi_old_q = shape * el_phi_old;
      const double w = ip.weight * trans->Weight();

      for (int i = 0; i < nd; ++i) {
        double grad_mu_term = 0.0;
        double grad_phi_term = 0.0;
        for (int d = 0; d < dim; ++d) {
          grad_mu_term += grad_mu(d) * dshape(i, d);
          grad_phi_term += grad_phi(d) * dshape(i, d);
        }
        r_phi(dofs[i]) +=
            w * (opts.xi * (phi_q - phi_old_q) / opts.dt * shape(i) + grad_mu_term);
        r_mu(dofs[i]) +=
            w * (-mu_q * shape(i) + opts.sigma_lg * opts.delta * grad_phi_term +
                 opts.sigma_lg / opts.delta * DoubleWellPrime(phi_q) * shape(i));

        for (int j = 0; j < nd; ++j) {
          double stiffness = 0.0;
          for (int d = 0; d < dim; ++d) {
            stiffness += dshape(j, d) * dshape(i, d);
          }
          const double mass_ij = shape(j) * shape(i);
          e00(i, j) += w * opts.xi / opts.dt * mass_ij;
          e01(i, j) += w * stiffness;
          e10(i, j) +=
              w * (opts.sigma_lg * opts.delta * stiffness +
                   opts.sigma_lg / opts.delta * DoubleWellSecond(phi_q) * mass_ij);
          e11(i, j) += -w * mass_ij;
        }
      }
    }
    AddElementMatrix(j00, dofs, e00);
    AddElementMatrix(j01, dofs, e01);
    AddElementMatrix(j10, dofs, e10);
    AddElementMatrix(j11, dofs, e11);
  }

  mfem::Vector el_phi_bdr, el_phi_old_bdr, bshape;
  mfem::DenseMatrix b10;
  for (int be = 0; be < fes.GetNBE(); ++be) {
    mfem::ElementTransformation* trans = fes.GetBdrElementTransformation(be);
    if (trans->Attribute != 1) {
      continue;
    }
    const mfem::FiniteElement* fe = fes.GetBE(be);
    fes.GetBdrElementDofs(be, dofs);
    phi.GetSubVector(dofs, el_phi_bdr);
    phi_old.GetSubVector(dofs, el_phi_old_bdr);
    const int nd = fe->GetDof();
    bshape.SetSize(nd);
    b10.SetSize(nd);
    b10 = 0.0;
    const mfem::IntegrationRule& ir =
        mfem::IntRules.Get(fe->GetGeomType(), 2 * fe->GetOrder() + trans->OrderW() + 2);
    for (int q = 0; q < ir.GetNPoints(); ++q) {
      const mfem::IntegrationPoint& ip = ir.IntPoint(q);
      trans->SetIntPoint(&ip);
      fe->CalcShape(ip, bshape);
      const double phi_q = bshape * el_phi_bdr;
      const double phi_old_q = bshape * el_phi_old_bdr;
      const double w = ip.weight * trans->Weight();
      const double boundary_reaction =
          opts.zeta * opts.delta * (phi_q - phi_old_q) / opts.dt +
          wall_factor * WallEnergyPrime(phi_q);
      const double boundary_jac =
          opts.zeta * opts.delta / opts.dt + wall_factor * WallEnergySecond(phi_q);
      for (int i = 0; i < nd; ++i) {
        r_mu(dofs[i]) += w * boundary_reaction * bshape(i);
        for (int j = 0; j < nd; ++j) {
          b10(i, j) += w * boundary_jac * bshape(j) * bshape(i);
        }
      }
    }
    AddElementMatrix(j10, dofs, b10);
  }

  j00.Finalize();
  j01.Finalize();
  j10.Finalize();
  j11.Finalize();
}

std::unique_ptr<mfem::HypreParMatrix> ParallelAssembleMatrix(
    mfem::ParFiniteElementSpace& fes, mfem::SparseMatrix& local_matrix) {
  mfem::ParBilinearForm form(&fes);
  return std::unique_ptr<mfem::HypreParMatrix>(form.ParallelAssemble(&local_matrix));
}

std::unique_ptr<mfem::HypreParMatrix> AssemblePhiPreconditioner(
    mfem::ParFiniteElementSpace& fes, const Options& opts) {
  const double wall_factor = (4.0 / (3.0 * std::sqrt(2.0))) * SigmaSLMinusSG(opts);
  mfem::ConstantCoefficient mass_coeff(opts.xi / opts.dt + opts.sigma_lg / opts.delta);
  mfem::ConstantCoefficient diffusion_coeff(opts.sigma_lg * opts.delta);
  mfem::ConstantCoefficient bdr_coeff(opts.zeta * opts.delta / opts.dt +
                                      1.5 * std::abs(wall_factor));
  mfem::ParBilinearForm form(&fes);
  form.AddDomainIntegrator(new mfem::MassIntegrator(mass_coeff));
  form.AddDomainIntegrator(new mfem::DiffusionIntegrator(diffusion_coeff));
  mfem::Array<int> bdr_marker;
  if (fes.GetParMesh()->bdr_attributes.Size() > 0) {
    bdr_marker.SetSize(fes.GetParMesh()->bdr_attributes.Max());
    bdr_marker = 0;
    bdr_marker[0] = 1;
    form.AddBoundaryIntegrator(new mfem::MassIntegrator(bdr_coeff), bdr_marker);
  }
  form.Assemble();
  form.Finalize();
  return std::unique_ptr<mfem::HypreParMatrix>(form.ParallelAssemble());
}

std::unique_ptr<mfem::HypreParMatrix> AssembleMassPreconditioner(
    mfem::ParFiniteElementSpace& fes) {
  mfem::ConstantCoefficient one(1.0);
  mfem::ParBilinearForm form(&fes);
  form.AddDomainIntegrator(new mfem::MassIntegrator(one));
  form.Assemble();
  form.Finalize();
  return std::unique_ptr<mfem::HypreParMatrix>(form.ParallelAssemble());
}

class ScaledSolver : public mfem::Solver {
 public:
  ScaledSolver(std::unique_ptr<mfem::Solver> inner, double scale)
      : mfem::Solver(inner->Height()), inner_(std::move(inner)), scale_(scale) {}

  void SetOperator(const mfem::Operator& op) override { MFEM_CONTRACT_VAR(op); }

  void Mult(const mfem::Vector& x, mfem::Vector& y) const override {
    inner_->Mult(x, y);
    y *= scale_;
  }

 private:
  std::unique_ptr<mfem::Solver> inner_;
  double scale_;
};

void AddSparseBlock(mfem::SparseMatrix& target, const mfem::SparseMatrix& block,
                    int row_offset, int col_offset) {
  const int* rows = block.HostReadI();
  const int* cols = block.HostReadJ();
  const double* data = block.HostReadData();
  for (int i = 0; i < block.Height(); ++i) {
    for (int k = rows[i]; k < rows[i + 1]; ++k) {
      target.Add(row_offset + i, col_offset + cols[k], data[k]);
    }
  }
}

std::unique_ptr<mfem::SparseMatrix> BuildMonolithicJacobian(
    const mfem::SparseMatrix& j00, const mfem::SparseMatrix& j01,
    const mfem::SparseMatrix& j10, const mfem::SparseMatrix& j11) {
  const int n = j00.Height();
  auto monolithic = std::make_unique<mfem::SparseMatrix>(2 * n);
  AddSparseBlock(*monolithic, j00, 0, 0);
  AddSparseBlock(*monolithic, j01, 0, n);
  AddSparseBlock(*monolithic, j10, n, 0);
  AddSparseBlock(*monolithic, j11, n, n);
  monolithic->Finalize();
  return monolithic;
}

class CachedDirectPreconditioner : public mfem::Solver {
 public:
  CachedDirectPreconditioner(int size, int rebuild_iteration_limit)
      : mfem::Solver(size), rebuild_iteration_limit_(rebuild_iteration_limit) {}

  void SetOperator(const mfem::Operator& op) override { MFEM_CONTRACT_VAR(op); }

  void RebuildIfNeeded(const mfem::SparseMatrix& j00, const mfem::SparseMatrix& j01,
                       const mfem::SparseMatrix& j10, const mfem::SparseMatrix& j11) {
    if (!NeedsRebuild()) {
      return;
    }
    monolithic_matrix_ = BuildMonolithicJacobian(j00, j01, j10, j11);
    solver_ = std::make_unique<mfem::UMFPackSolver>();
    solver_->SetPrintLevel(0);
#ifdef MFEM_USE_SUITESPARSE
    solver_->Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_METIS;
#endif
    solver_->SetOperator(*monolithic_matrix_);
    rebuild_next_solve_ = false;
    ++rebuild_count_;
  }

  void NoteOuterIterations(int iterations) {
    if (rebuild_iteration_limit_ > 0 && iterations > rebuild_iteration_limit_) {
      rebuild_next_solve_ = true;
    }
  }

  int RebuildCount() const { return rebuild_count_; }

  void Mult(const mfem::Vector& x, mfem::Vector& y) const override {
    MFEM_VERIFY(solver_ != nullptr, "Direct preconditioner has not been factorized.");
    solver_->Mult(x, y);
  }

 private:
  bool NeedsRebuild() const { return solver_ == nullptr || rebuild_next_solve_; }

  int rebuild_iteration_limit_ = 50;
  int rebuild_count_ = 0;
  bool rebuild_next_solve_ = true;
  std::unique_ptr<mfem::SparseMatrix> monolithic_matrix_;
  std::unique_ptr<mfem::UMFPackSolver> solver_;
};

double IntegrateGridFunction(mfem::ParFiniteElementSpace& fes, const mfem::ParGridFunction& gf) {
  double local_value = 0.0;
  mfem::Array<int> dofs;
  mfem::Vector el_gf, shape;
  for (int e = 0; e < fes.GetNE(); ++e) {
    const mfem::FiniteElement* fe = fes.GetFE(e);
    mfem::ElementTransformation* trans = fes.GetElementTransformation(e);
    fes.GetElementDofs(e, dofs);
    gf.GetSubVector(dofs, el_gf);
    shape.SetSize(fe->GetDof());
    const mfem::IntegrationRule& ir =
        mfem::IntRules.Get(fe->GetGeomType(), 2 * fe->GetOrder() + trans->OrderW());
    for (int q = 0; q < ir.GetNPoints(); ++q) {
      const mfem::IntegrationPoint& ip = ir.IntPoint(q);
      trans->SetIntPoint(&ip);
      fe->CalcShape(ip, shape);
      local_value += ip.weight * trans->Weight() * (shape * el_gf);
    }
  }
  double value = 0.0;
  MPI_Allreduce(&local_value, &value, 1, MPI_DOUBLE, MPI_SUM, fes.GetComm());
  return value;
}

double L2Difference(mfem::ParFiniteElementSpace& fes, const mfem::ParGridFunction& a,
                    const mfem::ParGridFunction& b) {
  double local_value = 0.0;
  mfem::Array<int> dofs;
  mfem::Vector ea, eb, shape;
  for (int e = 0; e < fes.GetNE(); ++e) {
    const mfem::FiniteElement* fe = fes.GetFE(e);
    mfem::ElementTransformation* trans = fes.GetElementTransformation(e);
    fes.GetElementDofs(e, dofs);
    a.GetSubVector(dofs, ea);
    b.GetSubVector(dofs, eb);
    shape.SetSize(fe->GetDof());
    const mfem::IntegrationRule& ir =
        mfem::IntRules.Get(fe->GetGeomType(), 2 * fe->GetOrder() + trans->OrderW());
    for (int q = 0; q < ir.GetNPoints(); ++q) {
      const mfem::IntegrationPoint& ip = ir.IntPoint(q);
      trans->SetIntPoint(&ip);
      fe->CalcShape(ip, shape);
      const double diff = shape * ea - shape * eb;
      local_value += ip.weight * trans->Weight() * diff * diff;
    }
  }
  double value = 0.0;
  MPI_Allreduce(&local_value, &value, 1, MPI_DOUBLE, MPI_SUM, fes.GetComm());
  return std::sqrt(value);
}

double MaxWallTime(double local_time, MPI_Comm comm) {
  double max_time = 0.0;
  MPI_Allreduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, comm);
  return max_time;
}

double TotalEnergy(mfem::ParFiniteElementSpace& fes, const mfem::ParGridFunction& phi,
                   const Options& opts) {
  double local_bulk = 0.0;
  mfem::Array<int> dofs;
  mfem::Vector el_phi, shape, grad_phi;
  mfem::DenseMatrix dshape;
  for (int e = 0; e < fes.GetNE(); ++e) {
    const mfem::FiniteElement* fe = fes.GetFE(e);
    mfem::ElementTransformation* trans = fes.GetElementTransformation(e);
    fes.GetElementDofs(e, dofs);
    phi.GetSubVector(dofs, el_phi);
    const int dim = fe->GetDim();
    shape.SetSize(fe->GetDof());
    dshape.SetSize(fe->GetDof(), dim);
    grad_phi.SetSize(dim);
    const mfem::IntegrationRule& ir =
        mfem::IntRules.Get(fe->GetGeomType(), 2 * fe->GetOrder() + trans->OrderW() + 2);
    for (int q = 0; q < ir.GetNPoints(); ++q) {
      const mfem::IntegrationPoint& ip = ir.IntPoint(q);
      trans->SetIntPoint(&ip);
      fe->CalcShape(ip, shape);
      fe->CalcPhysDShape(*trans, dshape);
      dshape.MultTranspose(el_phi, grad_phi);
      const double phi_q = shape * el_phi;
      local_bulk += ip.weight * trans->Weight() *
                    (0.5 * opts.delta * (grad_phi * grad_phi) + DoubleWell(phi_q) / opts.delta);
    }
  }
  double bulk = 0.0;
  MPI_Allreduce(&local_bulk, &bulk, 1, MPI_DOUBLE, MPI_SUM, fes.GetComm());
  bulk *= (3.0 * std::sqrt(2.0) / 4.0) * opts.sigma_lg;

  double local_wall = 0.0;
  mfem::Vector bshape, el_phi_bdr;
  for (int be = 0; be < fes.GetNBE(); ++be) {
    mfem::ElementTransformation* trans = fes.GetBdrElementTransformation(be);
    if (trans->Attribute != 1) {
      continue;
    }
    const mfem::FiniteElement* fe = fes.GetBE(be);
    fes.GetBdrElementDofs(be, dofs);
    phi.GetSubVector(dofs, el_phi_bdr);
    bshape.SetSize(fe->GetDof());
    const mfem::IntegrationRule& ir =
        mfem::IntRules.Get(fe->GetGeomType(), 2 * fe->GetOrder() + trans->OrderW() + 2);
    for (int q = 0; q < ir.GetNPoints(); ++q) {
      const mfem::IntegrationPoint& ip = ir.IntPoint(q);
      trans->SetIntPoint(&ip);
      fe->CalcShape(ip, bshape);
      local_wall += ip.weight * trans->Weight() * WallEnergy(bshape * el_phi_bdr);
    }
  }
  double wall = 0.0;
  MPI_Allreduce(&local_wall, &wall, 1, MPI_DOUBLE, MPI_SUM, fes.GetComm());
  return bulk + SigmaSLMinusSG(opts) * wall;
}

int NewtonCHStep(mfem::ParFiniteElementSpace& fes, mfem::ParGridFunction& phi,
                 mfem::ParGridFunction& mu, const mfem::ParGridFunction& phi_old,
                 const Options& opts, CachedDirectPreconditioner* direct_preconditioner,
                 int& max_linear_iterations, int& pc_rebuilds) {
  const int ndofs = fes.GetVSize();
  const int true_ndofs = fes.GetTrueVSize();
  mfem::Array<int> offsets(3);
  offsets[0] = 0;
  offsets[1] = true_ndofs;
  offsets[2] = true_ndofs;
  offsets.PartialSum();

  max_linear_iterations = 0;
  pc_rebuilds = 0;
  const int direct_rebuild_count_before =
      direct_preconditioner == nullptr ? 0 : direct_preconditioner->RebuildCount();
  for (int it = 1; it <= opts.newton_maxit; ++it) {
    mfem::Vector r0(ndofs), r1(ndofs);
    mfem::SparseMatrix j00(ndofs), j01(ndofs), j10(ndofs), j11(ndofs);
    AssembleCHResidualAndJacobian(fes, phi, mu, phi_old, opts, r0, r1, j00, j01, j10, j11);

    mfem::BlockVector rhs(offsets);
    rhs = 0.0;
    mfem::ParLinearForm r0_form(&fes);
    mfem::ParLinearForm r1_form(&fes);
    r0_form = r0;
    r1_form = r1;
    r0_form.ParallelAssemble(rhs.GetBlock(0));
    rhs.GetBlock(0).SyncAliasMemory(rhs);
    r1_form.ParallelAssemble(rhs.GetBlock(1));
    rhs.GetBlock(1).SyncAliasMemory(rhs);
    rhs *= -1.0;

    auto a00 = ParallelAssembleMatrix(fes, j00);
    auto a01 = ParallelAssembleMatrix(fes, j01);
    auto a10 = ParallelAssembleMatrix(fes, j10);
    auto a11 = ParallelAssembleMatrix(fes, j11);

    mfem::BlockOperator jacobian(offsets);
    jacobian.SetBlock(0, 0, a00.get());
    jacobian.SetBlock(0, 1, a01.get());
    jacobian.SetBlock(1, 0, a10.get());
    jacobian.SetBlock(1, 1, a11.get());

    mfem::BlockVector delta(offsets);
    delta = 0.0;
    mfem::GMRESSolver solver(fes.GetComm());
    solver.SetRelTol(opts.linear_tol);
    solver.SetAbsTol(opts.linear_tol);
    solver.SetMaxIter(opts.linear_maxit);
    solver.SetKDim(100);
    solver.SetPrintLevel(0);
    solver.SetOperator(jacobian);

    std::unique_ptr<mfem::HypreParMatrix> p_phi;
    std::unique_ptr<mfem::HypreParMatrix> p_mu;
    std::unique_ptr<mfem::HypreBoomerAMG> phi_amg;
    std::unique_ptr<mfem::HypreSmoother> mu_jacobi;
    std::unique_ptr<ScaledSolver> mu_solver;
    std::unique_ptr<mfem::BlockLowerTriangularPreconditioner> block_preconditioner;

    if (direct_preconditioner != nullptr) {
      direct_preconditioner->RebuildIfNeeded(j00, j01, j10, j11);
      solver.SetPreconditioner(*direct_preconditioner);
    } else {
      p_phi = AssemblePhiPreconditioner(fes, opts);
      p_mu = AssembleMassPreconditioner(fes);

      phi_amg = std::make_unique<mfem::HypreBoomerAMG>(*p_phi);
      phi_amg->SetPrintLevel(0);
      mu_jacobi = std::make_unique<mfem::HypreSmoother>(*p_mu, mfem::HypreSmoother::Jacobi);
      mu_solver = std::make_unique<ScaledSolver>(std::move(mu_jacobi), -1.0);

      block_preconditioner =
          std::make_unique<mfem::BlockLowerTriangularPreconditioner>(offsets);
      block_preconditioner->SetDiagonalBlock(0, phi_amg.get());
      block_preconditioner->SetBlock(1, 0, a10.get());
      block_preconditioner->SetDiagonalBlock(1, mu_solver.get());
      solver.SetPreconditioner(*block_preconditioner);
    }

    solver.Mult(rhs, delta);
    max_linear_iterations = std::max(max_linear_iterations, solver.GetNumIterations());
    if (direct_preconditioner != nullptr) {
      direct_preconditioner->NoteOuterIterations(solver.GetNumIterations());
      pc_rebuilds = direct_preconditioner->RebuildCount() - direct_rebuild_count_before;
    }
    if (!solver.GetConverged()) {
      return -1;
    }

    mfem::Vector local_delta_phi(ndofs);
    mfem::Vector local_delta_mu(ndofs);
    fes.GetProlongationMatrix()->Mult(delta.GetBlock(0), local_delta_phi);
    fes.GetProlongationMatrix()->Mult(delta.GetBlock(1), local_delta_mu);
    phi += local_delta_phi;
    mu += local_delta_mu;

    const double delta_norm =
        std::sqrt(mfem::InnerProduct(fes.GetComm(), delta.GetBlock(0), delta.GetBlock(0)) +
                  mfem::InnerProduct(fes.GetComm(), delta.GetBlock(1), delta.GetBlock(1)));
    if (delta_norm < opts.newton_tol) {
      return it;
    }
  }
  return -1;
}

void SaveParaview(mfem::ParaViewDataCollection* paraview, const Options& opts, int step,
                  double time, bool force = false) {
  if (paraview == nullptr || !opts.visualization ||
      (!force && (opts.output_every <= 0 || step % opts.output_every != 0))) {
    return;
  }
  paraview->SetCycle(step);
  paraview->SetTime(time);
  paraview->Save();
}

Options ParseOptions(int argc, char* argv[]) {
  Options opts;
  const bool output_dir_user_set = HasOutputDirectoryOption(argc, argv);
  const bool final_time_user_set = HasFinalTimeOption(argc, argv);
  mfem::OptionsParser args(argc, argv);
  args.AddOption(&opts.nx, "-nx", "--num-elements-x", "Number of elements in x.");
  args.AddOption(&opts.ny, "-ny", "--num-elements-y", "Number of elements in y.");
  args.AddOption(&opts.order, "-o", "--order", "Finite element order.");
  args.AddOption(&opts.theta_deg, "-theta", "--theta", "Young angle in degrees.");
  args.AddOption(&opts.delta, "-delta", "--delta", "Diffuse-interface thickness.");
  args.AddOption(&opts.dt, "-dt", "--time-step", "Target time step size.");
  args.AddOption(&opts.min_dt, "-dtmin", "--minimum-time-step",
                 "Smallest allowed adaptive retry time step.");
  args.AddOption(&opts.t_final, "-tf", "--t-final", "Final time.");
  args.AddOption(&opts.energy_decay_rate_tol, "-edrtol", "--energy-decay-rate-tol",
                 "Stop when (E^{n-1}-E^n)/dt is below this value. Negative disables stopping.");
  args.AddOption(&opts.newton_tol, "-ntol", "--newton-tol", "Newton tolerance.");
  args.AddOption(&opts.newton_maxit, "-nmax", "--newton-maxit", "Maximum Newton iterations.");
  args.AddOption(&opts.linear_tol, "-ltol", "--linear-tol",
                 "GMRES relative and absolute tolerance.");
  args.AddOption(&opts.linear_maxit, "-lmax", "--linear-maxit", "Maximum GMRES iterations.");
  args.AddOption(&opts.preconditioner_rebuild_iteration_limit, "-pcrebuild",
                 "--preconditioner-rebuild-iterations",
                 "For -pc direct, refactor before the next solve if GMRES takes more than this many iterations.");
  args.AddOption(&opts.preconditioner, "-pc", "--preconditioner",
                 "Linear preconditioner: block or direct. The direct option is serial-only.");
  args.AddOption(&opts.max_dt_retries, "-dt-retries", "--max-time-step-retries",
                 "Maximum number of step halvings after a failed Newton solve.");
  args.AddOption(&opts.output_every, "-oe", "--output-every", "ParaView output frequency.");
  args.AddOption(&opts.output_dir, "-out", "--output-dir", "Output directory.");
  args.AddOption(&opts.visualization, "-vis", "--visualization", "-no-vis",
                 "--no-visualization", "Enable or disable ParaView output.");
  args.AddOption(&opts.adaptive_time_step, "-adapt-dt", "--adaptive-time-step",
                 "-fixed-dt", "--fixed-time-step",
                 "Retry failed Newton steps with half the time step.");
  args.Parse();
  if (!args.Good()) {
    if (mfem::Mpi::WorldRank() == 0) {
      args.PrintUsage(std::cout);
    }
    std::exit(1);
  }
  if (!final_time_user_set) {
    opts.t_final = DefaultFinalTime(opts);
  }
  if (!output_dir_user_set) {
    opts.output_dir = DefaultOutputDirectory(opts);
  }
  if (opts.preconditioner != "block" && opts.preconditioner != "direct") {
    if (mfem::Mpi::WorldRank() == 0) {
      std::cerr << "Unknown preconditioner '" << opts.preconditioner
                << "'. Use 'block' or 'direct'.\n";
    }
    std::exit(1);
  }
  return opts;
}

}  // namespace

int main(int argc, char* argv[]) {
  mfem::Mpi::Init(argc, argv);
  mfem::Hypre::Init();
  const int myid = mfem::Mpi::WorldRank();
  MPI_Comm comm = MPI_COMM_WORLD;

  Options opts = ParseOptions(argc, argv);
  if (opts.preconditioner == "direct" && mfem::Mpi::WorldSize() != 1) {
    if (myid == 0) {
      std::cerr << "The UMFPACK direct preconditioner is serial-only in this MFEM build. "
                << "Run without mpirun, or use -pc block for MPI runs.\n";
    }
    MPI_Abort(comm, 1);
  }
  if (myid == 0) {
    std::filesystem::create_directories(opts.output_dir);
  }
  MPI_Barrier(comm);

  mfem::H1_FECollection fec(opts.order, 2);
  mfem::Mesh serial_mesh =
      mfem::Mesh::MakeCartesian2D(opts.nx, opts.ny, mfem::Element::QUADRILATERAL, false,
                                  opts.x_max - opts.x_min, opts.y_max - opts.y_min, false);
  ShiftMeshToDomain(serial_mesh, opts);
  mfem::ParMesh mesh(comm, serial_mesh);
  serial_mesh.Clear();
  mfem::ParFiniteElementSpace fes(&mesh, &fec);

  mfem::ParGridFunction phi(&fes);
  mfem::ParGridFunction phi_old(&fes);
  mfem::ParGridFunction mu(&fes);
  mfem::ParGridFunction mu_old(&fes);
  RectangularDropletCoefficient initial(opts);
  phi.ProjectCoefficient(initial);
  phi_old = phi;
  mu = 0.0;
  mu_old = 0.0;

  std::unique_ptr<CachedDirectPreconditioner> direct_preconditioner;
  if (opts.preconditioner == "direct") {
    direct_preconditioner = std::make_unique<CachedDirectPreconditioner>(
        2 * fes.GetTrueVSize(), opts.preconditioner_rebuild_iteration_limit);
  }

  std::unique_ptr<mfem::ParaViewDataCollection> paraview;
  if (opts.visualization) {
    paraview = std::make_unique<mfem::ParaViewDataCollection>("ConservativeWettingCH", &mesh);
    paraview->SetPrefixPath(opts.output_dir);
    paraview->SetDataFormat(mfem::VTKFormat::BINARY);
    paraview->SetHighOrderOutput(true);
    paraview->RegisterField("phi", &phi);
    paraview->RegisterField("mu", &mu);
  }

  std::ofstream history;
  if (myid == 0) {
    history.open(opts.output_dir + "/time_specialized.csv");
    history << "time,step,dt,dt_retries,newton_iterations,max_linear_iterations,pc_rebuilds,"
            << "mass,energy,energy_decay_rate,l2_update,step_wall_time,newton_wall_time\n";
  }

  double time = 0.0;
  int step = 0;
  double energy_old = TotalEnergy(fes, phi, opts);
  const double mass_initial = IntegrateGridFunction(fes, phi);

  if (myid == 0) {
    history << std::setprecision(16) << time << "," << step << ",0,0,0,0,0,"
            << mass_initial << "," << energy_old << ",nan,nan,0,0\n";
  }

  double current_dt = opts.dt;
  while (time < opts.t_final - 1.0e-14) {
    const double step_start = MPI_Wtime();
    ++step;
    current_dt = std::min(current_dt, opts.t_final - time);
    double pinned_time = 0.0;
    if (NextPinnedOutputTime(time, time + current_dt, pinned_time)) {
      current_dt = pinned_time - time;
    }

    int max_linear_iterations = 0;
    int pc_rebuilds = 0;
    int nit = -1;
    int dt_retries = 0;
    bool accepted = false;
    double newton_wall_time = 0.0;
    Options step_opts = opts;

    while (!accepted) {
      step_opts.dt = current_dt;
      phi = phi_old;
      mu = mu_old;
      max_linear_iterations = 0;
      pc_rebuilds = 0;
      const double newton_start = MPI_Wtime();
      nit = NewtonCHStep(fes, phi, mu, phi_old, step_opts, direct_preconditioner.get(),
                         max_linear_iterations, pc_rebuilds);
      newton_wall_time = MaxWallTime(MPI_Wtime() - newton_start, comm);
      accepted = nit > 0;
      if (accepted) {
        break;
      }

      phi = phi_old;
      mu = mu_old;
      if (!opts.adaptive_time_step || dt_retries >= opts.max_dt_retries ||
          0.5 * current_dt < opts.min_dt) {
        if (myid == 0) {
          std::cerr << "NewtonCHStep did not converge at time=" << time
                    << " step=" << step << " dt=" << current_dt
                    << ". Adaptive retry limit reached.\n";
        }
        MPI_Abort(comm, 1);
      }
      current_dt *= 0.5;
      ++dt_retries;
      if (myid == 0) {
        std::cout << "Newton retry: time=" << time << " step=" << step
                  << " reducing dt to " << current_dt << "\n";
      }
    }

    time += current_dt;

    const double mass = IntegrateGridFunction(fes, phi);
    const double energy = TotalEnergy(fes, phi, opts);
    const double rate = (energy_old - energy) / current_dt;
    const double update = L2Difference(fes, phi, phi_old);
    const double step_wall_time = MaxWallTime(MPI_Wtime() - step_start, comm);
    if (myid == 0) {
      history << std::setprecision(16) << time << "," << step << "," << current_dt << ","
              << dt_retries << "," << nit << "," << max_linear_iterations << ","
              << pc_rebuilds << ","
              << mass << "," << energy << "," << rate << "," << update << ","
              << step_wall_time << "," << newton_wall_time << "\n";

      std::cout << "step=" << step << " time=" << time << " nit=" << nit
                << " mass=" << mass << " energy=" << energy << " rate=" << rate
                << " update=" << update << " dt=" << current_dt
                << " dt_retries=" << dt_retries << " max_lit=" << max_linear_iterations
                << " pc_rebuilds=" << pc_rebuilds
                << " step_wall=" << step_wall_time << " newton_wall=" << newton_wall_time
                << "\n";
    }

    phi_old = phi;
    mu_old = mu;
    energy_old = energy;
    SaveParaview(paraview.get(), opts, step, time, IsPinnedOutputTime(time));

    if (opts.energy_decay_rate_tol >= 0.0 && rate < opts.energy_decay_rate_tol) {
      SaveParaview(paraview.get(), opts, step, time, true);
      if (myid == 0) {
        std::cout << "Stopping: energy decay rate " << rate << " < "
                  << opts.energy_decay_rate_tol << "\n";
      }
      break;
    }
    current_dt = std::min(opts.dt, 2.0 * current_dt);
  }

  SaveParaview(paraview.get(), opts, step, time, true);
  MPI_Finalize();
  return 0;
}
