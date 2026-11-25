#ifndef WETTING_2D_BENCHMARK_HPP
#define WETTING_2D_BENCHMARK_HPP

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <string>

#include "mfem.hpp"

namespace wetting_benchmark {

constexpr double kPi = 3.141592653589793238462643383279502884;

// The full parabola or circle extends through the wall. Measuring distance to
// this extension gives a smooth phase-field profile across the contact point.
class SmoothDropletCoefficient : public mfem::Coefficient {
 public:
  SmoothDropletCoefficient(double half_area, double theta_deg, double delta, bool circular)
      : circular_(circular), scale_(std::sqrt(2.0) * delta) {
    const double theta = theta_deg * kPi / 180.0;
    if (circular_) {
      radius_ = std::sqrt(2.0 * half_area / (theta - std::sin(theta) * std::cos(theta)));
      height_ = radius_ * (1.0 - std::cos(theta));
      width_ = theta >= 0.5 * kPi ? radius_ : radius_ * std::sin(theta);
    } else {
      width_ = std::sqrt(3.0 * half_area / std::tan(theta));
      height_ = 3.0 * half_area / (2.0 * width_);
      curvature_factor_ = height_ / (width_ * width_);
    }
  }

  double Width() const { return width_; }
  double Height() const { return height_; }

  mfem::real_t Eval(mfem::ElementTransformation& trans,
                    const mfem::IntegrationPoint& ip) override {
    mfem::Vector point;
    trans.Transform(ip, point);
    const double x = std::abs(point(0));
    const double y = point(1);
    if (circular_) {
      return std::tanh((std::hypot(x, y + radius_ - height_) - radius_) / scale_);
    }

    const double k = curvature_factor_;
    const double vertical_offset = height_ - y;
    // The derivative of squared distance is twice
    // 2 k^2 s^3 + (1 - 2 k (H-y)) s - x. For x>0 it has one positive
    // root. Its monotone part is bounded by the two limits below.
    const double cubic = 2.0 * k * k;
    const double linear = 1.0 - 2.0 * k * vertical_offset;
    double lower = std::sqrt(std::max(0.0, -linear / (3.0 * cubic)));
    double upper = std::max(x, std::sqrt(std::max(0.0, vertical_offset / k)));
    for (int iteration = 0; iteration < 42; ++iteration) {
      const double middle = 0.5 * (lower + upper);
      if ((cubic * middle * middle + linear) * middle < x) {
        lower = middle;
      } else {
        upper = middle;
      }
    }
    const double s = 0.5 * (lower + upper);
    const double distance = std::hypot(s - x, vertical_offset - k * s * s);
    const double sign = y < height_ - k * x * x ? -1.0 : 1.0;
    return sign * std::tanh(distance / scale_);
  }

 private:
  bool circular_;
  double scale_;
  double radius_ = 0.0;
  double width_ = 0.0;
  double height_ = 0.0;
  double curvature_factor_ = 0.0;
};

// Adjust the appropriate bulk phase toward zero with a smooth taper and
// unchanged sign. The taper occupies a fixed band at resolved mesh sizes,
// and no Q1 element cut by the zero contour is changed.
inline void MatchMassPreservingContour(mfem::ParFiniteElementSpace& fes,
                                       mfem::ParGridFunction& phi, double mass_error,
                                       double cell_diagonal, double delta) {
  if (std::abs(mass_error) <= 1.0e-13) {
    return;
  }
  mfem::ConstantCoefficient one(1.0);
  mfem::ParLinearForm mass_form(&fes);
  mass_form.AddDomainIntegrator(new mfem::DomainLFIntegrator(one));
  mass_form.Assemble();
  mfem::Vector mass_weights(fes.GetTrueVSize());
  mass_form.ParallelAssemble(mass_weights);

  mfem::Vector phase;
  phi.GetTrueDofs(phase);
  mfem::Vector capacity(phase.Size());
  const double sign = mass_error > 0.0 ? 1.0 : -1.0;
  const double start_distance = std::max(1.25 * delta, cell_diagonal);
  const double scale = std::sqrt(2.0) * delta;
  const double lower = std::tanh(start_distance / scale);
  const double upper = std::tanh((start_distance + delta) / scale);
  MFEM_VERIFY(upper > lower, "The diffuse interface is too thin to match the initial mass.");
  for (int i = 0; i < phase.Size(); ++i) {
    const double magnitude = -sign * phase(i);
    const double t = std::clamp((magnitude - lower) / (upper - lower), 0.0, 1.0);
    capacity(i) = magnitude * t * t * (3.0 - 2.0 * t);
  }
  const double available = mfem::InnerProduct(fes.GetComm(), mass_weights, capacity);
  MFEM_VERIFY(std::abs(mass_error) < available,
              "Cannot match the initial mass outside the diffuse interface; refine the mesh.");
  phase.Add(mass_error / available, capacity);
  phi.SetFromTrueDofs(phase);
}

class GeometryHistory {
 public:
  GeometryHistory(mfem::ParFiniteElementSpace& fes, double delta, bool enabled,
                  const std::string& output_dir)
      : fes_(fes), profile_scale_(std::sqrt(2.0) * delta), enabled_(enabled) {
    if (!enabled_) {
      return;
    }
    MPI_Comm_rank(fes_.GetComm(), &rank_);
    for (int b = 0; b < fes_.GetNBE(); ++b) {
      const int attr = fes_.GetParMesh()->GetBdrAttribute(b);
      if (attr == 1 || attr == 4) {
        edges_.Append(b);
      }
    }
    if (rank_ == 0) {
      file_.open(output_dir + "/geometry.csv");
      MFEM_VERIFY(file_.is_open(), "Cannot open 2D geometry history.");
      file_ << "time,step,x_cl,theta_cl_deg,apex_y,wall_profile_scale\n" << std::setprecision(16);
    }
  }

  void Write(const mfem::ParGridFunction& phi, int step, double time) {
    if (!enabled_) {
      return;
    }
    const double missing = -std::numeric_limits<double>::infinity();
    const double nan = std::numeric_limits<double>::quiet_NaN();
    struct LocatedValue {
      double value;
      int rank;
    };
    LocatedValue local[2] = {{missing, rank_}, {missing, rank_}};
    double wall[2] = {nan, nan};
    mfem::IntegrationPoint first, last, root;
    first.Set1w(0.0, 1.0);
    last.Set1w(1.0, 1.0);
    mfem::Vector point, gradient;
    for (int b : edges_) {
      auto* trans = fes_.GetBdrElementTransformation(b);
      const double a = phi.GetValue(*trans, first);
      const double c = phi.GetValue(*trans, last);
      if ((a <= 0.0) == (c <= 0.0)) {
        continue;
      }
      root.Set1w(a / (a - c), 1.0);
      trans->Transform(root, point);
      if (trans->Attribute == 1 && point(0) > local[0].value) {
        local[0].value = point(0);
        trans->SetIntPoint(&root);
        phi.GetGradient(*trans, gradient);
        const double magnitude = gradient.Norml2();
        wall[0] = std::acos(std::clamp(gradient(1) / magnitude, -1.0, 1.0)) * 180.0 / kPi;
        wall[1] = profile_scale_ * magnitude;
      } else if (trans->Attribute == 4) {
        local[1].value = std::max(local[1].value, point(1));
      }
    }

    LocatedValue global[2];
    MPI_Allreduce(local, global, 2, MPI_DOUBLE_INT, MPI_MAXLOC, fes_.GetComm());
    MPI_Bcast(wall, 2, MPI_DOUBLE, global[0].rank, fes_.GetComm());
    if (rank_ == 0) {
      const double footprint = std::isfinite(global[0].value) ? global[0].value : nan;
      const double apex = std::isfinite(global[1].value) ? global[1].value : nan;
      file_ << time << ',' << step << ',' << footprint << ',' << wall[0] << ','
            << apex << ',' << wall[1] << '\n';
    }
  }

 private:
  mfem::ParFiniteElementSpace& fes_;
  double profile_scale_;
  bool enabled_;
  int rank_ = 0;
  mfem::Array<int> edges_;
  std::ofstream file_;
};

}  // namespace wetting_benchmark

#endif
