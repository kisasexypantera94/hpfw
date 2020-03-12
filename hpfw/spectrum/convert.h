#pragma once

#include <Eigen/Dense>

namespace hpfw::spectrum {

    template<typename Real, int Rows, int Cols>
    auto power_to_db(const Eigen::Matrix<Real, Rows, Cols> &spectro) -> Eigen::Matrix<Real, Rows, Cols> {
        double maxVal = std::max(1e-10, double(spectro.maxCoeff()));
        Eigen::Matrix<Real, Rows, Cols> log_spec =
                10.0 * (spectro.array() < 1e-10).select(1e-10, spectro).array().log10()
                - 10.0 * log10(maxVal);

        maxVal = log_spec.maxCoeff();
        return (log_spec.array() < (maxVal - 80.0)).select(maxVal - 80.0, log_spec);
    }

    template<typename Real, int Rows, int Cols>
    auto amplitude_to_db(Eigen::Matrix<Real, Rows, Cols> &spectro) -> Eigen::Matrix<Real, Rows, Cols> {
        Eigen::Matrix<Real, Rows, Cols> spectro_p2 = spectro.unaryExpr([](const double x) {
            return std::pow(x, 2);
        });

        return power_to_db(spectro_p2);
    }

} // hpfw::spectrum