use crate::grid_function::GridFunction;
use crate::numerical_differentiation;

pub fn get_effective_field(
    exchange_constant: f64,
    dmi_constant: f64,
    magnetization: &GridFunction,
) -> GridFunction {
    exchange_effective_field(exchange_constant, magnetization)
        + dmi_effective_field(dmi_constant, magnetization)
}

fn exchange_effective_field(
    exchange_constant: f64,
    magnetization: &GridFunction,
) -> GridFunction {
    let laplacian_of_magnetization = magnetization.derivative().derivative();
    exchange_constant * laplacian_of_magnetization
}

fn dmi_effective_field(
    dmi_constant: f64,
    magnetization: &GridFunction,
) -> GridFunction {
    let magnetization_derivative = magnetization.derivative();
    let magnetization_second_derivative = magnetization_derivative.derivative();
    dmi_constant
        * magnetization_derivative
            .cross_product(&magnetization_second_derivative)
}
