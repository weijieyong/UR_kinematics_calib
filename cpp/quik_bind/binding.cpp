#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>
#include "quik/Robot.hpp"
#include "quik/IKSolver.hpp"
#include <memory>

namespace py = pybind11;
using Eigen::VectorXd;
using Eigen::Matrix4d;

// Global instances for robot and IK solver
static std::shared_ptr<quik::Robot<>> global_robot = nullptr;
static std::shared_ptr<quik::IKSolver<>> global_solver = nullptr;

PYBIND11_MODULE(quik_bind, m) {
    m.doc() = "Python bindings for quik inverse kinematics and forward kinematics";

    // Initialize robot and solver with DH parameters, link types, optional Qsign, Tbase, Ttool
    m.def("init_robot", [](const Eigen::MatrixXd& dh,
                           const Eigen::Matrix<bool, Eigen::Dynamic, 1>& linkTypes,
                           const Eigen::VectorXd& qsign,
                           const Matrix4d& Tbase,
                           const Matrix4d& Ttool) {
        // Convert boolean linkTypes to JOINTTYPE_t enum
        Eigen::Matrix<quik::JOINTTYPE_t, Eigen::Dynamic, 1> lt_enum(linkTypes.size());
        for(int i = 0; i < linkTypes.size(); ++i)
            lt_enum(i) = linkTypes(i) ? quik::JOINT_PRISMATIC : quik::JOINT_REVOLUTE;
        // Handle default qsign
        Eigen::VectorXd qsign_vec = qsign.size() == 0 ? Eigen::VectorXd::Ones(linkTypes.size()) : qsign;
        global_robot = std::make_shared<quik::Robot<>>(dh, lt_enum, qsign_vec, Tbase, Ttool);
        global_solver = std::make_shared<quik::IKSolver<>>(global_robot);
    },
    py::arg("dh"), py::arg("linkTypes"), py::arg("qsign") = Eigen::VectorXd(),
    py::arg("Tbase") = Matrix4d::Identity(), py::arg("Ttool") = Matrix4d::Identity(),
    "Initialize the robot model with DH table, link types, optional Qsign, base and tool transforms");

    m.def("fkn", [](const VectorXd& q, int frame) {
        // Assume user initialized a global Robot and IKSolver
        Matrix4d T;
        global_robot->FKn(q, T, frame);
        return T;
    }, py::arg("q"), py::arg("frame") = -1,
    "Compute forward kinematics for joint vector q and return frame transform");

    m.def("ik", [](const Matrix4d& twt, const VectorXd& q0, int max_iter) {
        VectorXd q_star = q0;
        Eigen::Matrix<double,6,1> e_star;
        int iter;
        quik::BREAKREASON_t br;
        global_solver->IK(twt, q0, q_star, e_star, iter, br);
        return py::make_tuple(q_star, e_star, iter, quik::breakreason2str(br));
    }, py::arg("twt"), py::arg("q0"), py::arg("max_iter") = 100,
    "Solve inverse kinematics for target transform twt and initial q0");
}
