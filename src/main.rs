pub mod grid;
pub mod quadratic_interpolation;
pub mod scalar_field;
pub mod vector_field;
// pub mod landau_lifshitz_gilbert_equation;

use grid::Grid;
use scalar_field::ScalarField1D;
use vector_field::VectorField1D;

fn main() {
    let grid = Grid::new_uniform_grid(0.0, 1.0, 20);

    let scalar_field = ScalarField1D::function_to_scalar_field(&grid, |x| x);
    let partial_x_scalar_field = scalar_field.partial_x();

    let expected_result = ScalarField1D::new_constant_scalar_field(&grid, 1.0);

    println!("{:?}", partial_x_scalar_field.field_values);
    println!("{:?}", expected_result.field_values);
}
