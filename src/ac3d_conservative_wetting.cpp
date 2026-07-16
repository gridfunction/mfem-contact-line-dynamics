/**
 * @file main.cpp
 * @brief 3D conservative Allen--Cahn wetting/dewetting benchmark with contact-line relaxation.
 *
 * This is a first MFEM implementation of the 3D scheme used in the droplet manuscript.
 * The nonlinear residual and scalar Newton block are assembled explicitly.  The volume
 * constraint is enforced by a one-dimensional Schur complement:
 *
 *   A u = -R_phi,  A q = m,
 *   delta_lambda = (-R_c - m^T u)/(m^T q),
 *   delta_phi = u + delta_lambda q.
 */

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "mfem.hpp"  // NOLINT [no include directory when naming mfem include file]

namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;

struct Options {
  int nx = 128;
  int ny = 128;
  int nz = 128;
  int order = 1;
  double x_min = 0.0;
  double x_max = 1.0;
  double y_min = 0.0;
  double y_max = 1.0;
  double z_min = 0.0;
  double z_max = 1.0;
  double droplet_x_min = 0.0;
  double droplet_x_max = 0.5;
  double droplet_y_min = 0.0;
  double droplet_y_max = 0.5;
  double droplet_z_min = 0.0;
  double droplet_z_max = 0.25;
  double theta_deg = 45.0;
  double delta = 0.01;
  double dt = 0.01;
  double min_dt = 1.0e-4;
  double t_final = 10.0;
  double energy_decay_rate_tol = 1.0e-7;
  double sigma_lg = 1.0;
  double xi = 1.0;
  double zeta = 1.0;
  double convex_splitting_until = -1.0;
  double newton_tol = 1.0e-10;
  int newton_maxit = 20;
  double linear_tol = 1.0e-8;
  int linear_maxit = 500;
  int preconditioner_rebuild_iteration_limit = 30;
  int gmg_levels = 2;
  int max_dt_retries = 8;
  int output_every = 10;
  std::string preconditioner = "amg";
  std::string output_dir;
  bool visualization = true;
  bool smooth_initial_condition = false;
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

std::string DefaultOutputDirectory(const Options& opts) {
  std::ostringstream os;
  os << "Saves/ac3d_conservative_wetting"
     << "_theta" << NumberToken(opts.theta_deg)
     << "_nx" << opts.nx
     << "_ny" << opts.ny
     << "_nz" << opts.nz
     << "_delta" << NumberToken(opts.delta)
     << "_dt" << NumberToken(opts.dt);
  if (opts.convex_splitting_until > 0.0) {
    os << "_csuntil" << NumberToken(opts.convex_splitting_until);
  }
  if (opts.preconditioner == "gmg") {
    os << "_gmg";
  }
  return os.str();
}

double DomainVolume(const Options& opts) {
  return (opts.x_max - opts.x_min) * (opts.y_max - opts.y_min) * (opts.z_max - opts.z_min);
}

double SharpInitialMass(const Options& opts) {
  const double droplet_volume =
      (opts.droplet_x_max - opts.droplet_x_min) *
      (opts.droplet_y_max - opts.droplet_y_min) *
      (opts.droplet_z_max - opts.droplet_z_min);
  return DomainVolume(opts) - 2.0 * droplet_volume;
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
                                  (opts_.y_max - opts_.y_min) / opts_.ny) *
        BoundaryAnchoredIndicator(xx(2), opts_.z_min, opts_.droplet_z_min,
                                  opts_.droplet_z_max,
                                  (opts_.z_max - opts_.z_min) / opts_.nz);
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

class SmoothRectangularDropletCoefficient : public mfem::Coefficient {
 public:
  explicit SmoothRectangularDropletCoefficient(const Options& opts) : opts_(opts) {}

  double Eval(mfem::ElementTransformation& trans,
              const mfem::IntegrationPoint& ip) override {
    mfem::Vector xx;
    trans.Transform(ip, xx);
    const double dx = std::max(xx(0) - opts_.droplet_x_max, 0.0);
    const double dy = std::max(xx(1) - opts_.droplet_y_max, 0.0);
    const double dz = std::max(xx(2) - opts_.droplet_z_max, 0.0);
    const double outside_distance = std::sqrt(dx * dx + dy * dy + dz * dz);
    const bool inside =
        xx(0) <= opts_.droplet_x_max &&
        xx(1) <= opts_.droplet_y_max &&
        xx(2) <= opts_.droplet_z_max;
    const double signed_distance = inside
        ? -std::min({opts_.droplet_x_max - xx(0), opts_.droplet_y_max - xx(1),
                     opts_.droplet_z_max - xx(2)})
        : outside_distance;
    return std::tanh(signed_distance / (std::sqrt(2.0) * opts_.delta));
  }

 private:
  const Options& opts_;
};

void ShiftMeshToDomain(mfem::Mesh& mesh, const Options& opts) {
  for (int i = 0; i < mesh.GetNV(); ++i) {
    double* vertex = mesh.GetVertex(i);
    vertex[0] += opts.x_min;
    vertex[1] += opts.y_min;
    vertex[2] += opts.z_min;
  }
}

bool IsSubstrateBoundary(mfem::ElementTransformation& trans, const Options& opts) {
  const mfem::IntegrationRule& ir = mfem::IntRules.Get(trans.GetGeometryType(), 1);
  const mfem::IntegrationPoint& ip = ir.IntPoint(0);
  trans.SetIntPoint(&ip);
  mfem::Vector xx;
  trans.Transform(ip, xx);
  return std::abs(xx(2) - opts.z_min) < 1.0e-12;
}

void AssembleMassVector(mfem::ParFiniteElementSpace& fes, mfem::Vector& mass_vec) {
  const int ndofs = fes.GetVSize();
  mass_vec.SetSize(ndofs);
  mass_vec = 0.0;

  mfem::Array<int> dofs;
  mfem::Vector shape;
  for (int e = 0; e < fes.GetNE(); ++e) {
    const mfem::FiniteElement* fe = fes.GetFE(e);
    mfem::ElementTransformation* trans = fes.GetElementTransformation(e);
    fes.GetElementDofs(e, dofs);
    const int nd = fe->GetDof();
    shape.SetSize(nd);
    const mfem::IntegrationRule& ir =
        mfem::IntRules.Get(fe->GetGeomType(), 2 * fe->GetOrder() + trans->OrderW());
    for (int q = 0; q < ir.GetNPoints(); ++q) {
      const mfem::IntegrationPoint& ip = ir.IntPoint(q);
      trans->SetIntPoint(&ip);
      fe->CalcShape(ip, shape);
      const double w = ip.weight * trans->Weight();
      for (int i = 0; i < nd; ++i) {
        mass_vec(dofs[i]) += w * shape(i);
      }
    }
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

void AssembleResidualAndJacobian(mfem::ParFiniteElementSpace& fes,
                                 const mfem::ParGridFunction& phi,
                                 const mfem::ParGridFunction& phi_old, const Options& opts,
                                 double lambda, bool use_convex_splitting,
                                 mfem::Vector& residual, mfem::SparseMatrix& jacobian) {
  const int ndofs = fes.GetVSize();
  residual.SetSize(ndofs);
  residual = 0.0;

  const double wall_factor = (4.0 / (3.0 * std::sqrt(2.0))) * SigmaSLMinusSG(opts);
  mfem::Array<int> dofs;
  mfem::Vector el_phi, el_phi_old, shape, grad_phi;
  mfem::DenseMatrix dshape, elmat;

  for (int e = 0; e < fes.GetNE(); ++e) {
    const mfem::FiniteElement* fe = fes.GetFE(e);
    mfem::ElementTransformation* trans = fes.GetElementTransformation(e);
    fes.GetElementDofs(e, dofs);
    phi.GetSubVector(dofs, el_phi);
    phi_old.GetSubVector(dofs, el_phi_old);

    const int nd = fe->GetDof();
    const int dim = fe->GetDim();
    shape.SetSize(nd);
    dshape.SetSize(nd, dim);
    grad_phi.SetSize(dim);
    elmat.SetSize(nd);
    elmat = 0.0;

    const mfem::IntegrationRule& ir =
        mfem::IntRules.Get(fe->GetGeomType(), 2 * fe->GetOrder() + trans->OrderW() + 2);
    for (int q = 0; q < ir.GetNPoints(); ++q) {
      const mfem::IntegrationPoint& ip = ir.IntPoint(q);
      trans->SetIntPoint(&ip);
      fe->CalcShape(ip, shape);
      fe->CalcPhysDShape(*trans, dshape);
      dshape.MultTranspose(el_phi, grad_phi);

      const double phi_q = shape * el_phi;
      const double phi_old_q = shape * el_phi_old;
      const double w = ip.weight * trans->Weight();
      const double reaction = use_convex_splitting
          ? opts.xi * opts.delta * (phi_q - phi_old_q) / opts.dt +
                opts.sigma_lg / opts.delta * (phi_q * phi_q * phi_q - phi_old_q) - lambda
          : opts.xi * opts.delta * (phi_q - phi_old_q) / opts.dt +
                opts.sigma_lg / opts.delta * DoubleWellPrime(phi_q) - lambda;
      const double reaction_jac = use_convex_splitting
          ? opts.xi * opts.delta / opts.dt + opts.sigma_lg / opts.delta * 3.0 * phi_q * phi_q
          : opts.xi * opts.delta / opts.dt +
                opts.sigma_lg / opts.delta * DoubleWellSecond(phi_q);

      for (int i = 0; i < nd; ++i) {
        double grad_term = 0.0;
        for (int d = 0; d < dim; ++d) {
          grad_term += grad_phi(d) * dshape(i, d);
        }
        residual(dofs[i]) += w * (reaction * shape(i) + opts.sigma_lg * opts.delta * grad_term);
        for (int j = 0; j < nd; ++j) {
          double stiffness = 0.0;
          for (int d = 0; d < dim; ++d) {
            stiffness += dshape(j, d) * dshape(i, d);
          }
          elmat(i, j) += w * (reaction_jac * shape(j) * shape(i) +
                              opts.sigma_lg * opts.delta * stiffness);
        }
      }
    }
    AddElementMatrix(jacobian, dofs, elmat);
  }

  mfem::Vector el_phi_bdr, el_phi_old_bdr, bshape;
  mfem::DenseMatrix bdr_elmat;
  for (int be = 0; be < fes.GetNBE(); ++be) {
    mfem::ElementTransformation* trans = fes.GetBdrElementTransformation(be);
    if (!IsSubstrateBoundary(*trans, opts)) {
      continue;  // The other cube faces are natural symmetry/no-flux boundaries.
    }
    const mfem::FiniteElement* fe = fes.GetBE(be);
    fes.GetBdrElementDofs(be, dofs);
    phi.GetSubVector(dofs, el_phi_bdr);
    phi_old.GetSubVector(dofs, el_phi_old_bdr);
    const int nd = fe->GetDof();
    bshape.SetSize(nd);
    bdr_elmat.SetSize(nd);
    bdr_elmat = 0.0;
    const mfem::IntegrationRule& ir =
        mfem::IntRules.Get(fe->GetGeomType(), 2 * fe->GetOrder() + trans->OrderW() + 2);
    for (int q = 0; q < ir.GetNPoints(); ++q) {
      const mfem::IntegrationPoint& ip = ir.IntPoint(q);
      trans->SetIntPoint(&ip);
      fe->CalcShape(ip, bshape);
      const double phi_q = bshape * el_phi_bdr;
      const double phi_old_q = bshape * el_phi_old_bdr;
      const double w = ip.weight * trans->Weight();
      const double wall_lipschitz = 1.5 * std::abs(wall_factor);
      const double boundary_reaction =
          opts.zeta * opts.delta * (phi_q - phi_old_q) / opts.dt +
          wall_factor * WallEnergyPrime(phi_q) +
          (use_convex_splitting ? wall_lipschitz * (phi_q - phi_old_q) : 0.0);
      const double boundary_jac =
          opts.zeta * opts.delta / opts.dt + wall_factor * WallEnergySecond(phi_q) +
          (use_convex_splitting ? wall_lipschitz : 0.0);
      for (int i = 0; i < nd; ++i) {
        residual(dofs[i]) += w * boundary_reaction * bshape(i);
        for (int j = 0; j < nd; ++j) {
          bdr_elmat(i, j) += w * boundary_jac * bshape(j) * bshape(i);
        }
      }
    }
    AddElementMatrix(jacobian, dofs, bdr_elmat);
  }
  jacobian.Finalize();
}

class FixedPreconditioner : public mfem::Solver {
 public:
  explicit FixedPreconditioner(std::unique_ptr<mfem::Solver> inner)
      : mfem::Solver(inner->Height()), inner_(std::move(inner)) {}

  void SetOperator(const mfem::Operator& op) override {
    MFEM_CONTRACT_VAR(op);
  }

  void Mult(const mfem::Vector& x, mfem::Vector& y) const override {
    inner_->Mult(x, y);
  }

 private:
  std::unique_ptr<mfem::Solver> inner_;
};

class CachedLinearSolver {
 public:
  CachedLinearSolver(std::string preconditioner_type, int rebuild_iteration_limit,
                     const mfem::ParFiniteElementSpaceHierarchy* hierarchy)
      : preconditioner_type_(std::move(preconditioner_type)),
        rebuild_iteration_limit_(rebuild_iteration_limit),
        hierarchy_(hierarchy) {}

  int Solve(const mfem::HypreParMatrix& mat, const mfem::Vector& rhs, mfem::Vector& sol,
            const Options& opts, double tol, int maxit, bool use_gmres) {
    sol.SetSize(rhs.Size());
    sol = 0.0;
    const std::string requested_type = preconditioner_type_ == "gmg" ? "gmg" : "amg";

    if (NeedsRebuild(mat, requested_type)) {
      Rebuild(mat, opts, requested_type);
    }

    int iterations = 0;
    const bool use_krylov_gmres = use_gmres || requested_type == "gmg";
    if (use_krylov_gmres) {
      mfem::GMRESSolver solver(MPI_COMM_WORLD);
      solver.SetRelTol(tol);
      solver.SetAbsTol(tol);
      solver.SetMaxIter(maxit);
      solver.SetKDim(50);
      solver.SetPrintLevel(0);
      solver.SetPreconditioner(*preconditioner_);
      solver.SetOperator(mat);
      solver.Mult(rhs, sol);
      iterations = solver.GetNumIterations();
    } else {
      mfem::CGSolver solver(MPI_COMM_WORLD);
      solver.SetRelTol(tol);
      solver.SetAbsTol(tol);
      solver.SetMaxIter(maxit);
      solver.SetPrintLevel(0);
      solver.SetPreconditioner(*preconditioner_);
      solver.SetOperator(mat);
      solver.Mult(rhs, sol);
      iterations = solver.GetNumIterations();
    }
    if (rebuild_iteration_limit_ > 0 && iterations > rebuild_iteration_limit_) {
      rebuild_next_solve_ = true;
    }
    return iterations;
  }

  int RebuildCount() const { return rebuild_count_; }

  void Clear() {
    preconditioner_.reset();
    preconditioner_matrix_.reset();
    gmg_forms_.clear();
    active_preconditioner_type_.clear();
    preconditioner_size_ = 0;
    rebuild_next_solve_ = false;
  }

 private:
  bool NeedsRebuild(const mfem::HypreParMatrix& mat,
                    const std::string& requested_type) const {
    return !preconditioner_ || rebuild_next_solve_ || preconditioner_size_ != mat.Height() ||
           active_preconditioner_type_ != requested_type;
  }

  void Rebuild(const mfem::HypreParMatrix& mat, const Options& opts,
               const std::string& requested_type) {
    preconditioner_.reset();
    preconditioner_matrix_.reset();
    gmg_forms_.clear();
    preconditioner_size_ = mat.Height();
    if (requested_type == "gmg") {
      BuildGMG(mat, opts);
    } else {
      BuildAMG(mat);
    }
    active_preconditioner_type_ = requested_type;
    rebuild_next_solve_ = false;
    ++rebuild_count_;
  }

  void BuildAMG(const mfem::HypreParMatrix& mat) {
    preconditioner_matrix_ = std::make_unique<mfem::HypreParMatrix>(mat);
    auto amg = std::make_unique<mfem::HypreBoomerAMG>(*preconditioner_matrix_);
    amg->SetPrintLevel(0);
    preconditioner_ = std::move(amg);
  }

  static void BuildApproximateLevel(const mfem::ParFiniteElementSpace& level_fes,
                                    const Options& opts, mfem::Operator*& op,
                                    mfem::Solver*& smoother,
                                    bool use_amg_coarse_solver,
                                    std::vector<std::unique_ptr<mfem::ParBilinearForm>>& forms) {
    mfem::ConstantCoefficient mass_coeff(opts.xi * opts.delta / opts.dt +
                                         opts.sigma_lg / opts.delta);
    mfem::ConstantCoefficient diffusion_coeff(opts.sigma_lg * opts.delta);
    auto form = std::make_unique<mfem::ParBilinearForm>(
        const_cast<mfem::ParFiniteElementSpace*>(&level_fes));
    form->AddDomainIntegrator(new mfem::MassIntegrator(mass_coeff));
    form->AddDomainIntegrator(new mfem::DiffusionIntegrator(diffusion_coeff));
    form->Assemble();
    mfem::Array<int> empty_tdofs;
    mfem::OperatorPtr opr;
    opr.SetType(mfem::Operator::Hypre_ParCSR);
    form->FormSystemMatrix(empty_tdofs, opr);
    opr.SetOperatorOwner(false);

    op = opr.Ptr();
    if (use_amg_coarse_solver) {
      auto* coarse_mat = dynamic_cast<mfem::HypreParMatrix*>(op);
      MFEM_VERIFY(coarse_mat != nullptr, "GMG coarse operator must be a HypreParMatrix.");
      auto* coarse_amg = new mfem::HypreBoomerAMG(*coarse_mat);
      coarse_amg->SetPrintLevel(0);
      smoother = coarse_amg;
    } else {
      mfem::Vector diag(level_fes.GetTrueVSize());
      form->AssembleDiagonal(diag);
      smoother = new mfem::OperatorChebyshevSmoother(
          *opr, diag, empty_tdofs, 2, level_fes.GetParMesh()->GetComm());
    }
    forms.emplace_back(std::move(form));
  }

  void BuildGMG(const mfem::HypreParMatrix& mat, const Options& opts) {
    if (hierarchy_ == nullptr || hierarchy_->GetNumLevels() < 2) {
      BuildAMG(mat);
      return;
    }

    const int nlevels = hierarchy_->GetNumLevels();
    mfem::Array<mfem::Operator*> operators(nlevels);
    mfem::Array<mfem::Solver*> smoothers(nlevels);
    mfem::Array<mfem::Operator*> prolongations(nlevels - 1);
    mfem::Array<bool> owned_operators(nlevels);
    mfem::Array<bool> owned_smoothers(nlevels);
    mfem::Array<bool> owned_prolongations(nlevels - 1);

    for (int level = 0; level < nlevels - 1; ++level) {
      auto& level_fes = hierarchy_->GetFESpaceAtLevel(level);
      BuildApproximateLevel(level_fes, opts, operators[level], smoothers[level],
                            level == 0, gmg_forms_);
      // The ParBilinearForm storage owns the HypreParMatrix returned by
      // FormSystemMatrix.  Multigrid only borrows it.
      owned_operators[level] = false;
      owned_smoothers[level] = true;
      prolongations[level] = hierarchy_->GetProlongationAtLevel(level);
      owned_prolongations[level] = false;
    }

    auto& fine_fes = hierarchy_->GetFESpaceAtLevel(nlevels - 1);
    BuildApproximateLevel(fine_fes, opts, operators[nlevels - 1],
                          smoothers[nlevels - 1], false, gmg_forms_);
    owned_operators[nlevels - 1] = false;
    owned_smoothers[nlevels - 1] = true;

    auto mg = std::make_unique<mfem::Multigrid>(
        operators, smoothers, prolongations, owned_operators, owned_smoothers,
        owned_prolongations);
    mg->SetCycleType(mfem::Multigrid::CycleType::VCYCLE, 1, 1);
    preconditioner_ = std::make_unique<FixedPreconditioner>(std::move(mg));
    preconditioner_matrix_.reset();
  }

  std::string preconditioner_type_;
  std::string active_preconditioner_type_;
  int rebuild_iteration_limit_;
  int preconditioner_size_ = 0;
  int rebuild_count_ = 0;
  bool rebuild_next_solve_ = false;
  const mfem::ParFiniteElementSpaceHierarchy* hierarchy_ = nullptr;
  std::vector<std::unique_ptr<mfem::ParBilinearForm>> gmg_forms_;
  std::unique_ptr<mfem::HypreParMatrix> preconditioner_matrix_;
  std::unique_ptr<mfem::Solver> preconditioner_;
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

void CorrectMassToSharpValueBounded(mfem::ParFiniteElementSpace& fes,
                                    mfem::ParGridFunction& gf, const Options& opts) {
  mfem::Vector local_mass_vec;
  AssembleMassVector(fes, local_mass_vec);
  mfem::ParLinearForm mass_form(&fes);
  mass_form = local_mass_vec;
  mfem::Vector mass_vec(fes.GetTrueVSize());
  mass_form.ParallelAssemble(mass_vec);

  mfem::Vector true_phi;
  gf.GetTrueDofs(true_phi);
  const double target_mass = SharpInitialMass(opts);
  for (int pass = 0; pass < 6; ++pass) {
    const double current_mass = IntegrateGridFunction(fes, gf);
    const double remaining = target_mass - current_mass;
    if (std::abs(remaining) < 10.0 * std::numeric_limits<double>::epsilon()) {
      break;
    }

    double local_capacity = 0.0;
    const bool use_fractional_only = pass == 0;
    for (int i = 0; i < true_phi.Size(); ++i) {
      if (use_fractional_only && std::abs(true_phi(i)) > 1.0 - 1.0e-12) {
        continue;
      }
      const double capacity = remaining > 0.0 ? 1.0 - true_phi(i) : true_phi(i) + 1.0;
      if (capacity > 0.0) {
        local_capacity += mass_vec(i) * capacity;
      }
    }
    double global_capacity = 0.0;
    MPI_Allreduce(&local_capacity, &global_capacity, 1, MPI_DOUBLE, MPI_SUM, fes.GetComm());
    if (global_capacity <= std::abs(remaining)) {
      continue;
    }

    const double scale = remaining / global_capacity;
    for (int i = 0; i < true_phi.Size(); ++i) {
      if (use_fractional_only && std::abs(true_phi(i)) > 1.0 - 1.0e-12) {
        continue;
      }
      const double capacity = remaining > 0.0 ? 1.0 - true_phi(i) : true_phi(i) + 1.0;
      if (capacity > 0.0) {
        true_phi(i) += scale * capacity;
      }
    }
    gf.SetFromTrueDofs(true_phi);
  }
  gf.SetFromTrueDofs(true_phi);
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
    if (!IsSubstrateBoundary(*trans, opts)) {
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

int NewtonSchurStep(mfem::ParFiniteElementSpace& fes, mfem::ParGridFunction& phi,
                    const mfem::ParGridFunction& phi_old, const mfem::Vector& mass_vec,
                    const Options& opts, bool use_convex_splitting, double& lambda,
                    CachedLinearSolver& linear_solver, int& max_linear_iterations,
                    int& pc_rebuilds) {
  const int ndofs = fes.GetVSize();
  const int true_ndofs = fes.GetTrueVSize();
  max_linear_iterations = 0;
  const int rebuild_count_before = linear_solver.RebuildCount();
  for (int it = 1; it <= opts.newton_maxit; ++it) {
    mfem::Vector residual(ndofs);
    mfem::SparseMatrix jacobian(ndofs);
    AssembleResidualAndJacobian(fes, phi, phi_old, opts, lambda, use_convex_splitting,
                                residual, jacobian);

    mfem::ParLinearForm residual_form(&fes);
    residual_form = residual;
    mfem::Vector true_residual(true_ndofs);
    residual_form.ParallelAssemble(true_residual);

    mfem::ParBilinearForm jacobian_form(&fes);
    std::unique_ptr<mfem::HypreParMatrix> true_jacobian(jacobian_form.ParallelAssemble(&jacobian));

    mfem::ParGridFunction diff(&fes);
    diff = phi;
    diff -= phi_old;
    const double constraint_residual = IntegrateGridFunction(fes, diff);

    mfem::Vector rhs(true_residual);
    rhs *= -1.0;
    mfem::Vector u_corr, q_corr;
    const int u_iterations =
        linear_solver.Solve(*true_jacobian, rhs, u_corr, opts, opts.linear_tol,
                            opts.linear_maxit, !use_convex_splitting);
    const int q_iterations =
        linear_solver.Solve(*true_jacobian, mass_vec, q_corr, opts, opts.linear_tol,
                            opts.linear_maxit, !use_convex_splitting);
    max_linear_iterations =
        std::max(max_linear_iterations, std::max(u_iterations, q_iterations));

    const double schur = mfem::InnerProduct(fes.GetComm(), mass_vec, q_corr);
    const double delta_lambda =
        (-constraint_residual - mfem::InnerProduct(fes.GetComm(), mass_vec, u_corr)) / schur;
    mfem::Vector delta_phi(u_corr);
    delta_phi.Add(delta_lambda, q_corr);

    mfem::Vector local_delta_phi(ndofs);
    fes.GetProlongationMatrix()->Mult(delta_phi, local_delta_phi);

    phi += local_delta_phi;
    lambda += delta_lambda;

    const double delta_norm = std::sqrt(mfem::InnerProduct(fes.GetComm(), delta_phi, delta_phi));
    if (delta_norm < opts.newton_tol && std::abs(constraint_residual) < opts.newton_tol) {
      pc_rebuilds = linear_solver.RebuildCount() - rebuild_count_before;
      return it;
    }
  }
  pc_rebuilds = linear_solver.RebuildCount() - rebuild_count_before;
  return -1;
}

void SaveParaview(mfem::ParaViewDataCollection* paraview, const Options& opts, int step,
                  double time, bool force = false) {
  if (paraview == nullptr || !opts.visualization || (!force && step % opts.output_every != 0)) {
    return;
  }
  paraview->SetCycle(step);
  paraview->SetTime(time);
  paraview->Save();
}

Options ParseOptions(int argc, char* argv[]) {
  Options opts;
  const bool output_dir_user_set = HasOutputDirectoryOption(argc, argv);
  mfem::OptionsParser args(argc, argv);
  args.AddOption(&opts.nx, "-nx", "--num-elements-x", "Number of elements in x.");
  args.AddOption(&opts.ny, "-ny", "--num-elements-y", "Number of elements in y.");
  args.AddOption(&opts.nz, "-nz", "--num-elements-z", "Number of elements in z.");
  args.AddOption(&opts.order, "-o", "--order", "Finite element order.");
  args.AddOption(&opts.theta_deg, "-theta", "--theta", "Young angle in degrees.");
  args.AddOption(&opts.delta, "-delta", "--delta", "Diffuse-interface thickness.");
  args.AddOption(&opts.dt, "-dt", "--time-step", "Time step size.");
  args.AddOption(&opts.min_dt, "-dtmin", "--minimum-time-step",
                 "Smallest allowed adaptive retry time step.");
  args.AddOption(&opts.t_final, "-tf", "--t-final", "Final time.");
  args.AddOption(&opts.energy_decay_rate_tol, "-edrtol", "--energy-decay-rate-tol",
                 "Stop when (E^{n-1}-E^n)/dt is below this value. Negative disables stopping.");
  args.AddOption(&opts.convex_splitting_until, "-cs-until", "--convex-splitting-until",
                 "Use convex splitting for steps starting before this physical time.");
  args.AddOption(&opts.newton_tol, "-ntol", "--newton-tol", "Newton tolerance.");
  args.AddOption(&opts.newton_maxit, "-nmax", "--newton-maxit", "Maximum Newton iterations.");
  args.AddOption(&opts.linear_tol, "-ltol", "--linear-tol",
                 "Krylov relative and absolute tolerance.");
  args.AddOption(&opts.linear_maxit, "-lmax", "--linear-maxit",
                 "Maximum Krylov iterations.");
  args.AddOption(&opts.preconditioner_rebuild_iteration_limit, "-pcrebuild",
                 "--preconditioner-rebuild-iterations",
                 "Rebuild the preconditioner before the next solve if the Krylov solve takes more than this many iterations.");
  args.AddOption(&opts.preconditioner, "-pc", "--preconditioner",
                 "Linear preconditioner: amg or gmg.");
  args.AddOption(&opts.gmg_levels, "-gmg-levels", "--geometric-multigrid-levels",
                 "Number of geometric multigrid levels, including the finest level.");
  args.AddOption(&opts.max_dt_retries, "-dt-retries", "--max-time-step-retries",
                 "Maximum number of step halvings after a failed Newton solve.");
  args.AddOption(&opts.output_every, "-oe", "--output-every", "ParaView output frequency.");
  args.AddOption(&opts.output_dir, "-out", "--output-dir", "Output directory.");
  args.AddOption(&opts.visualization, "-vis", "--visualization", "-no-vis",
                 "--no-visualization", "Enable or disable ParaView output.");
  args.AddOption(&opts.smooth_initial_condition, "-smooth-init", "--smooth-initial-condition",
                 "-sharp-init", "--sharp-initial-condition",
                 "Use a tanh-smoothed or sharp rectangular initial phase field.");
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
  if (!output_dir_user_set) {
    opts.output_dir = DefaultOutputDirectory(opts);
  }
  if (opts.preconditioner != "amg" && opts.preconditioner != "gmg") {
    if (mfem::Mpi::WorldRank() == 0) {
      std::cerr << "Unknown preconditioner '" << opts.preconditioner
                << "'. Use 'amg' or 'gmg'.\n";
    }
    std::exit(1);
  }
  if (opts.gmg_levels < 2) {
    opts.gmg_levels = 2;
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
  if (myid == 0) {
    std::filesystem::create_directories(opts.output_dir);
  }
  MPI_Barrier(comm);

  mfem::H1_FECollection fec(opts.order, 3);
  std::unique_ptr<mfem::ParMesh> direct_mesh;
  std::unique_ptr<mfem::ParFiniteElementSpace> direct_fes;
  std::unique_ptr<mfem::ParFiniteElementSpaceHierarchy> hierarchy;
  mfem::ParMesh* mesh_ptr = nullptr;
  mfem::ParFiniteElementSpace* fes_ptr = nullptr;

  if (opts.preconditioner == "gmg") {
    const int refinement_factor = 1 << (opts.gmg_levels - 1);
    if (opts.nx % refinement_factor != 0 || opts.ny % refinement_factor != 0 ||
        opts.nz % refinement_factor != 0) {
      if (myid == 0) {
        std::cerr << "For -pc gmg, nx, ny, and nz must be divisible by "
                  << refinement_factor << " for " << opts.gmg_levels
                  << " multigrid levels.\n";
      }
      MPI_Abort(comm, 1);
    }
    mfem::Mesh serial_coarse = mfem::Mesh::MakeCartesian3D(
        opts.nx / refinement_factor, opts.ny / refinement_factor,
        opts.nz / refinement_factor, mfem::Element::HEXAHEDRON,
        opts.x_max - opts.x_min, opts.y_max - opts.y_min, opts.z_max - opts.z_min,
        false);
    ShiftMeshToDomain(serial_coarse, opts);
    auto* coarse_mesh = new mfem::ParMesh(comm, serial_coarse);
    serial_coarse.Clear();
    auto* coarse_fes = new mfem::ParFiniteElementSpace(coarse_mesh, &fec);
    hierarchy = std::make_unique<mfem::ParFiniteElementSpaceHierarchy>(
        coarse_mesh, coarse_fes, true, true);
    for (int level = 1; level < opts.gmg_levels; ++level) {
      hierarchy->AddUniformlyRefinedLevel();
    }
    mesh_ptr = hierarchy->GetFinestFESpace().GetParMesh();
    fes_ptr = &hierarchy->GetFinestFESpace();
  } else {
    mfem::Mesh serial_mesh = mfem::Mesh::MakeCartesian3D(
        opts.nx, opts.ny, opts.nz, mfem::Element::HEXAHEDRON,
        opts.x_max - opts.x_min, opts.y_max - opts.y_min, opts.z_max - opts.z_min,
        false);
    ShiftMeshToDomain(serial_mesh, opts);
    direct_mesh = std::make_unique<mfem::ParMesh>(comm, serial_mesh);
    serial_mesh.Clear();
    direct_fes = std::make_unique<mfem::ParFiniteElementSpace>(direct_mesh.get(), &fec);
    mesh_ptr = direct_mesh.get();
    fes_ptr = direct_fes.get();
  }

  mfem::ParMesh& mesh = *mesh_ptr;
  mfem::ParFiniteElementSpace& fes = *fes_ptr;
  mfem::ParGridFunction phi(&fes);
  mfem::ParGridFunction phi_old(&fes);
  RectangularDropletCoefficient sharp_initial(opts);
  SmoothRectangularDropletCoefficient smooth_initial(opts);
  if (opts.smooth_initial_condition) {
    phi.ProjectCoefficient(smooth_initial);
  } else {
    phi.ProjectCoefficient(sharp_initial);
  }
  CorrectMassToSharpValueBounded(fes, phi, opts);
  phi_old = phi;

  std::unique_ptr<mfem::ParaViewDataCollection> paraview;
  if (opts.visualization) {
    paraview = std::make_unique<mfem::ParaViewDataCollection>("ConservativeWettingAC3D", &mesh);
    paraview->SetPrefixPath(opts.output_dir);
    paraview->SetDataFormat(mfem::VTKFormat::BINARY);
    paraview->SetHighOrderOutput(true);
    paraview->RegisterField("phi", &phi);
  }

  mfem::Vector local_mass_vec;
  AssembleMassVector(fes, local_mass_vec);
  mfem::ParLinearForm mass_form(&fes);
  mass_form = local_mass_vec;
  mfem::Vector mass_vec(fes.GetTrueVSize());
  mass_form.ParallelAssemble(mass_vec);
  CachedLinearSolver linear_solver(opts.preconditioner,
                                   opts.preconditioner_rebuild_iteration_limit,
                                   hierarchy.get());

  std::ofstream history;
  if (myid == 0) {
    history.open(opts.output_dir + "/time_specialized.csv");
    history << "time,step,dt,dt_retries,newton_iterations,max_linear_iterations,pc_rebuilds,"
            << "mass,energy,energy_decay_rate,l2_update,lambda,step_wall_time,"
            << "newton_wall_time\n";
  }

  double time = 0.0;
  int step = 0;
  double lambda = 0.0;
  double energy_old = TotalEnergy(fes, phi, opts);
  SaveParaview(paraview.get(), opts, step, time);
  const double mass_initial = IntegrateGridFunction(fes, phi);

  if (myid == 0) {
    history << std::setprecision(16) << time << "," << step << ",0,0,0,0,0,"
            << mass_initial << "," << energy_old << ",nan,nan," << lambda << ",0,0\n";
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
    const double lambda_old = lambda;
    int max_linear_iterations = 0;
    int pc_rebuilds = 0;
    int nit = -1;
    int dt_retries = 0;
    bool accepted = false;
    double newton_wall_time = 0.0;
    Options step_opts = opts;
    const bool use_convex_splitting =
        opts.convex_splitting_until > 0.0 && time < opts.convex_splitting_until - 1.0e-14;

    while (!accepted) {
      step_opts.dt = current_dt;
      phi = phi_old;
      lambda = lambda_old;
      max_linear_iterations = 0;
      pc_rebuilds = 0;
      const double newton_start = MPI_Wtime();
      nit = NewtonSchurStep(fes, phi, phi_old, mass_vec, step_opts, use_convex_splitting,
                            lambda, linear_solver, max_linear_iterations, pc_rebuilds);
      newton_wall_time = MaxWallTime(MPI_Wtime() - newton_start, comm);
      accepted = nit > 0;
      if (accepted) {
        break;
      }

      phi = phi_old;
      lambda = lambda_old;
      linear_solver.Clear();
      if (!opts.adaptive_time_step || dt_retries >= opts.max_dt_retries ||
          0.5 * current_dt < opts.min_dt) {
        if (myid == 0) {
          std::cerr << "NewtonSchurStep did not converge at time=" << time
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
              << pc_rebuilds << "," << mass << "," << energy << "," << rate << ","
              << update << "," << lambda << "," << step_wall_time << "," << newton_wall_time
              << "\n";

      std::cout << "step=" << step << " time=" << time << " nit=" << nit << " mass=" << mass
                << " energy=" << energy << " rate=" << rate << " update=" << update
                << " scheme=" << (use_convex_splitting ? "cs" : "fi")
                << " dt=" << current_dt << " dt_retries=" << dt_retries
                << " max_lit=" << max_linear_iterations << " pc_rebuilds=" << pc_rebuilds
                << " step_wall=" << step_wall_time << " newton_wall=" << newton_wall_time
                << "\n";
    }

    phi_old = phi;
    energy_old = energy;
    SaveParaview(paraview.get(), opts, step, time, IsPinnedOutputTime(time));

    if (opts.energy_decay_rate_tol >= 0.0 && rate < opts.energy_decay_rate_tol) {
      if (step % opts.output_every != 0) {
        SaveParaview(paraview.get(), opts, step, time, true);
      }
      if (myid == 0) {
        std::cout << "Stopping: energy decay rate " << rate << " < "
                  << opts.energy_decay_rate_tol << "\n";
      }
      break;
    }
    current_dt = std::min(opts.dt, 2.0 * current_dt);
  }

  SaveParaview(paraview.get(), opts, step, time, true);
  linear_solver.Clear();
  MPI_Finalize();
  return 0;
}
