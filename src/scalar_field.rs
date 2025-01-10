use crate::boundary_conditions::{self, BoundaryConditions1D};
use crate::grid::Grid;
use crate::quadratic_interpolation;

/// # Scalar field 1D
///
/// ## Description
/// A `ScalarField1D` represents a scalar field sampled on a 1D grid of points.
/// Each grid point has an associated scalar, stored as an `f64`. The grid
/// points are represented by a `Grid` instance.
///
/// The x axis is the axis along which the grid points lie.
///
/// The scalar field can be subject to boundary conditions, which are specified
/// as a `BoundaryConditions1D` instance.
///
#[derive(Debug, Clone, PartialEq)]
pub struct ScalarField1D {
    pub grid: Grid,
    pub field_values: Vec<f64>,
    pub boundary_conditions: BoundaryConditions1D,
}

// Constructor functions.
impl ScalarField1D {
    /// # New scalar field
    ///
    /// ## Description
    /// Creates a new 1D scalar field, with a specified scalar at each grid
    /// point.
    ///
    pub fn new_scalar_field(
        grid: &Grid,
        field_values: &Vec<f64>,
    ) -> Result<Self, &'static str> {
        if field_values.len() != grid.grid_points.len() {
            return Err(
                "Number of field values does not match number of grid points",
            );
        }

        let mut scalar_field = ScalarField1D {
            grid: grid.clone(),
            field_values: field_values.clone(),
            boundary_conditions: None,
        };

        BoundaryConditions1D::check_scalar_bcs(&scalar_field, 1e-6)?;
        BoundaryConditions1D::apply_scalar_bcs(&mut scalar_field);
        Ok(scalar_field)
    }

    /// # Function to scalar field
    ///
    /// ## Description
    /// Generates a `ScalarField1D`, given a scalar valued function and a
    /// `Grid`. The function is sampled at each grid point and the function
    /// values are stored in the field `field_values`.
    ///
    pub fn function_to_scalar_field<F>(
        grid: &Grid,
        func: F,
        boundary_conditions: &BoundaryConditions1D,
    ) -> Result<Self, &'static str>
    where
        F: Fn(f64) -> f64,
    {
        let mut scalar_field = ScalarField1D {
            grid: grid.clone(),
            field_values: grid.grid_points.iter().map(|&x| func(x)).collect(),
            boundary_conditions: boundary_conditions.clone(),
        };

        BoundaryConditions1D::check_scalar_bcs(&scalar_field, 1e-6)?;
        BoundaryConditions1D::apply_scalar_bcs(&mut scalar_field);
        Ok(scalar_field)
    }

    /// # New constant scalar field
    ///
    /// ## Description
    /// Creates a new 1D scalar field with a constant value at every grid point.
    ///
    pub fn new_constant_scalar_field(
        grid: &Grid,
        field_value: f64,
        boundary_conditions: &BoundaryConditions1D,
    ) -> Result<Self, &'static str> {
        let mut scalar_field = ScalarField1D {
            grid: grid.clone(),
            field_values: vec![field_value; grid.grid_points.len()],
            boundary_conditions: boundary_conditions.clone(),
        };

        BoundaryConditions1D::check_scalar_bcs(&scalar_field, 1e-6)?;
        BoundaryConditions1D::apply_scalar_bcs(&mut scalar_field);
        Ok(scalar_field)
    }
}

// Arithmetic operations.
impl ScalarField1D {
    /// # Add scalar fields
    ///
    /// ## Description
    /// Adds `scalar_field` to the current scalar field, and returns this new
    /// scalar field.
    ///
    /// The boundary conditions of the new scalar field must be specified.
    ///
    pub fn add(
        &self,
        scalar_field: &ScalarField1D,
        boundary_conditions: &BoundaryConditions1D,
    ) -> Result<Self, &'static str> {
        if self.grid != scalar_field.grid {
            return Err("Grids do not match");
        }

        let field_values: Vec<f64> = self
            .field_values
            .iter()
            .zip(scalar_field.field_values.iter())
            .map(|(v1, v2)| v1 + v2)
            .collect();

        let mut scalar_field = ScalarField1D {
            grid: self.grid.clone(),
            field_values,
            boundary_conditions: boundary_conditions.clone(),
        };

        BoundaryConditions1D::check_scalar_bcs(&scalar_field, 1e-6)?;
        BoundaryConditions1D::apply_scalar_bcs(&mut scalar_field);
        Ok(scalar_field)
    }

    /// # Subtract scalar fields
    ///
    /// ## Description
    /// Subtracts `scalar_field` from the current scalar field, and returns this
    /// new scalar field.
    ///
    /// The boundary conditions of the new scalar field must be specified.
    ///
    pub fn subtract(
        &self,
        scalar_field: &ScalarField1D,
        boundary_conditions: &BoundaryConditions1D,
    ) -> Result<Self, &'static str> {
        if self.grid != scalar_field.grid {
            return Err("Grids do not match");
        }

        let field_values: Vec<f64> = self
            .field_values
            .iter()
            .zip(scalar_field.field_values.iter())
            .map(|(v1, v2)| v1 - v2)
            .collect();

        let mut scalar_field = ScalarField1D {
            grid: self.grid.clone(),
            field_values,
            boundary_conditions: boundary_conditions.clone(),
        };

        BoundaryConditions1D::check_scalar_bcs(&scalar_field, 1e-6)?;
        BoundaryConditions1D::apply_scalar_bcs(&mut scalar_field);
        Ok(scalar_field)
    }

    /// # Multiply scalar fields
    ///
    /// ## Description
    /// Multiplies the current scalar field by `scalar_field`, and returns this
    /// new scalar field.
    ///
    /// The boundary conditions of the new scalar field must be specified.
    ///
    pub fn multiply(
        &self,
        scalar_field: &ScalarField1D,
        boundary_conditions: &BoundaryConditions1D,
    ) -> Result<Self, &'static str> {
        if self.grid != scalar_field.grid {
            return Err("Grids do not match");
        }

        let field_values: Vec<f64> = self
            .field_values
            .iter()
            .zip(scalar_field.field_values.iter())
            .map(|(v1, v2)| v1 * v2)
            .collect();

        let mut scalar_field = ScalarField1D {
            grid: self.grid.clone(),
            field_values,
            boundary_conditions: boundary_conditions.clone(),
        };

        BoundaryConditions1D::check_scalar_bcs(&scalar_field, 1e-6)?;
        BoundaryConditions1D::apply_scalar_bcs(&mut scalar_field);
        Ok(scalar_field)
    }

    /// # Divide scalar fields
    ///
    /// ## Description
    /// Divides the current scalar field by `scalar_field`, and returns this new
    /// scalar field.
    ///
    /// The boundary conditions of the new scalar field must be specified.
    ///
    pub fn divide(
        &self,
        scalar_field: &ScalarField1D,
        boundary_conditions: &BoundaryConditions1D,
    ) -> Result<Self, &'static str> {
        if self.grid != scalar_field.grid {
            return Err("Grids do not match");
        }

        let field_values: Vec<f64> = self
            .field_values
            .iter()
            .zip(scalar_field.field_values.iter())
            .map(|(v1, v2)| {
                if *v2 == 0.0 {
                    return Err("Division by zero");
                }
                Ok(v1 / v2)
            })
            .collect::<Result<Vec<f64>, &'static str>>()?;

        let mut scalar_field = ScalarField1D {
            grid: self.grid.clone(),
            field_values,
            boundary_conditions: boundary_conditions.clone(),
        };

        BoundaryConditions1D::check_scalar_bcs(&scalar_field, 1e-6)?;
        BoundaryConditions1D::apply_scalar_bcs(&mut scalar_field);
        Ok(scalar_field)
    }

    /// # Scale scalar field.
    ///
    /// ## Description
    /// Multiplies the current scalar field by a scalar, and returns this new
    /// scalar field.
    ///
    /// The boundary conditions of the new scalar field must be specified.
    ///
    pub fn scale(
        &self,
        scalar: f64,
        boundary_conditions: &BoundaryConditions1D,
    ) -> Result<Self, &'static str> {
        let mut scalar_field = ScalarField1D {
            grid: self.grid.clone(),
            field_values: self
                .field_values
                .iter()
                .map(|v| v * scalar)
                .collect(),
            boundary_conditions: boundary_conditions.clone(),
        };

        BoundaryConditions1D::check_scalar_bcs(&scalar_field, 1e-6)?;
        BoundaryConditions1D::apply_scalar_bcs(&mut scalar_field);
        Ok(scalar_field)
    }
}

// Calculus operations.
impl ScalarField1D {
    /// # Partial derivative with respect to x
    ///
    /// ## Description
    /// Calculates the partial derivative of the scalar field with respect to
    /// x, and returns this result as a new scalar field.
    ///
    /// This derivative is calculated using the best differentiation scheme
    /// currently available.
    ///
    /// Different methods to handle the edge points are used for different
    /// boundary conditions.
    ///  
    pub fn partial_x(&self) -> Result<Self, &'static str> {
        match self.boundary_conditions {
            BoundaryConditions1D::DirichletScalar(_, _)
            | BoundaryConditions1D::None => {
                Ok(self.central_difference_derivative_dirichlet_bcs())
            }
            BoundaryConditions1D::Periodic => {
                Ok(self.central_difference_derivative_periodic_bcs())
            }
            BoundaryConditions1D::NeumannScalar(
                left_boundary,
                right_boundary,
            ) => Ok(self.central_difference_derivative_neumann_bcs(
                left_boundary,
                right_boundary,
            )),
            BoundaryConditions1D::DirichletVector(_, _)
            | BoundaryConditions1D::NeumannVector(_, _) => {
                return Err("Vector boundary conditions not compatible with scalar derivative")
            }
        }
    }

    /// # Central difference derivative dirichlet BCs
    ///
    /// ## Description
    /// Calculates the partial derivative of the scalar field with respect to x
    /// using the central difference scheme, and returns this result as a new
    /// scalar field.
    ///
    /// The derivative at the starting grid point is calculated using a     
    /// quadratic interpolation. The derivative at each interior grid point is
    /// calculated using the central difference scheme. The derivative at the
    /// final grid point is calculated using a quadratic interpolation.
    ///
    /// This function can calculate the derivative of a scalar field with
    /// None or Dirichlet boundary conditions.
    ///
    fn central_difference_derivative_dirichlet_bcs(self: &Self) -> Self {
        let grid = &self.grid;
        let field_values = &self.field_values;

        let grid_points = &grid.grid_points;
        let num_points = grid_points.len();

        let mut partial_x_values = Vec::with_capacity(num_points);

        // Calculates the derivative at the starting grid point using a
        // quadratic interpolation.
        let points = [grid_points[0], grid_points[1], grid_points[2]];
        let function_values =
            [field_values[0], field_values[1], field_values[2]];
        let coefficients =
            quadratic_interpolation::quadratic_interpolation_coefficients(
                points,
                function_values,
            )
            .unwrap();
        partial_x_values.push(quadratic_interpolation::quadratic_derivative(
            coefficients,
            grid_points[0],
        ));

        // Calculates the derivative at each interior grid point using the central difference scheme.
        for i in 1..(num_points - 1) {
            partial_x_values.push(
                (field_values[i + 1] - field_values[i - 1])
                    / (grid_points[i + 1] - grid_points[i - 1]),
            );
        }

        // Calculates the derivative at the final grid point using a quadratic
        // interpolation.
        let points = [
            grid_points[num_points - 3],
            grid_points[num_points - 2],
            grid_points[num_points - 1],
        ];
        let function_values = [
            field_values[num_points - 3],
            field_values[num_points - 2],
            field_values[num_points - 1],
        ];
        let coefficients =
            quadratic_interpolation::quadratic_interpolation_coefficients(
                points,
                function_values,
            )
            .unwrap();
        partial_x_values.push(quadratic_interpolation::quadratic_derivative(
            coefficients,
            grid_points[num_points - 1],
        ));

        ScalarField1D {
            grid: grid.clone(),
            field_values: partial_x_values,
            boundary_conditions: self.boundary_conditions.clone(),
        }
    }

    /// # Central difference derivative periodic BCs
    ///
    /// ## Description
    /// Calculates the partial derivative of the scalar field with respect to x
    /// using the central difference scheme, and returns this result as a new
    /// scalar field.
    ///
    /// The derivative at all grid points is calculated using the central
    /// difference scheme.
    ///
    /// This function can calculate the derivative of a scalar field with
    /// periodic boundary conditions.
    ///
    fn central_difference_derivative_periodic_bcs(self: &Self) -> Self {}

    /// # Central difference derivative Neumann BCs
    ///
    /// ## Description
    /// Calculates the partial derivative of the scalar field with respect to x
    /// using the central difference scheme, and returns this result as a new
    /// scalar field.
    ///
    /// The derivative at the interior grid points is calculated using the
    /// central difference scheme. The derivative at the edge points is known
    /// from the boundary conditions.
    ///
    /// This function can calculate the derivative of a scalar field with
    /// Neumann boundary conditions.
    ///
    fn central_difference_derivative_neumann_bcs(
        self: &Self,
        left_boundary: f64,
        right_boundary: f64,
    ) -> Self {
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::Grid;
    use crate::utils;

    #[test]
    fn test_new_scalar_field() {
        let grid = Grid::new_uniform_grid(0.0, 1.0, 5);
        let field_values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let scalar_field = ScalarField1D::new_scalar_field(
            &grid,
            &field_values,
            &BoundaryConditions1D::None,
        )
        .unwrap();
        assert_eq!(scalar_field.grid, grid);
        assert_eq!(scalar_field.field_values, field_values);
    }

    #[test]
    fn test_function_to_scalar_field() {
        let grid = Grid::new_uniform_grid(0.0, 1.0, 6);
        let scalar_field = ScalarField1D::function_to_scalar_field(
            &grid,
            |x| x,
            &BoundaryConditions1D::None,
        )
        .unwrap();

        let expected_values = vec![0.0, 0.2, 0.4, 0.6, 0.8, 1.0];
        let expected_scalar_field = ScalarField1D::new_scalar_field(
            &grid,
            &expected_values,
            &BoundaryConditions1D::None,
        )
        .unwrap();

        assert!(utils::scalar_field_equality(
            &expected_scalar_field,
            &scalar_field,
            1e-6
        ));
    }

    #[test]
    fn test_new_constant_scalar_field() {
        let grid = Grid::new_uniform_grid(0.0, 1.0, 5);
        let field_value = 5.0;
        let scalar_field = ScalarField1D::new_constant_scalar_field(
            &grid,
            field_value,
            &BoundaryConditions1D::None,
        )
        .unwrap();
        assert_eq!(scalar_field.grid, grid);
        assert_eq!(
            scalar_field.field_values,
            vec![field_value; grid.grid_points.len()]
        );
    }

    #[test]
    fn test_add() {
        let grid = Grid::new_uniform_grid(0.0, 1.0, 5);
        let field_values1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let field_values2 = vec![4.0, 3.0, 2.0, 1.0, 0.0];
        let scalar_field1 = ScalarField1D::new_scalar_field(
            &grid,
            &field_values1,
            &BoundaryConditions1D::None,
        )
        .unwrap();
        let scalar_field2 = ScalarField1D::new_scalar_field(
            &grid,
            &field_values2,
            &BoundaryConditions1D::None,
        )
        .unwrap();
        let result = scalar_field1
            .add(&scalar_field2, &BoundaryConditions1D::None)
            .unwrap();
        assert_eq!(result.field_values, vec![5.0, 5.0, 5.0, 5.0, 5.0]);
    }

    #[test]
    fn test_subtract() {
        let grid = Grid::new_uniform_grid(0.0, 1.0, 5);
        let field_values1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let field_values2 = vec![4.0, 3.0, 2.0, 1.0, 0.0];
        let scalar_field1 = ScalarField1D::new_scalar_field(
            &grid,
            &field_values1,
            &BoundaryConditions1D::None,
        )
        .unwrap();
        let scalar_field2 = ScalarField1D::new_scalar_field(
            &grid,
            &field_values2,
            &BoundaryConditions1D::None,
        )
        .unwrap();
        let result = scalar_field1
            .subtract(&scalar_field2, &BoundaryConditions1D::None)
            .unwrap();
        assert_eq!(result.field_values, vec![-3.0, -1.0, 1.0, 3.0, 5.0]);
    }

    #[test]
    fn test_multiply() {
        let grid = Grid::new_uniform_grid(0.0, 1.0, 5);
        let field_values1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let field_values2 = vec![4.0, 3.0, 2.0, 1.0, 0.0];
        let scalar_field1 = ScalarField1D::new_scalar_field(
            &grid,
            &field_values1,
            &BoundaryConditions1D::None,
        )
        .unwrap();
        let scalar_field2 = ScalarField1D::new_scalar_field(
            &grid,
            &field_values2,
            &BoundaryConditions1D::None,
        )
        .unwrap();
        let result = scalar_field1
            .multiply(&scalar_field2, &BoundaryConditions1D::None)
            .unwrap();
        assert_eq!(result.field_values, vec![4.0, 6.0, 6.0, 4.0, 0.0]);
    }

    #[test]
    fn test_divide() {
        let grid = Grid::new_uniform_grid(0.0, 1.0, 5);
        let field_values1 = vec![4.0, 9.0, 16.0, 25.0, 36.0];
        let field_values2 = vec![2.0, 3.0, 4.0, 5.0, 6.0];
        let scalar_field1 = ScalarField1D::new_scalar_field(
            &grid,
            &field_values1,
            &BoundaryConditions1D::None,
        )
        .unwrap();
        let scalar_field2 = ScalarField1D::new_scalar_field(
            &grid,
            &field_values2,
            &BoundaryConditions1D::None,
        )
        .unwrap();
        let result = scalar_field1
            .divide(&scalar_field2, &BoundaryConditions1D::None)
            .unwrap();
        assert_eq!(result.field_values, vec![2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_scale() {
        let grid = Grid::new_uniform_grid(0.0, 1.0, 5);
        let field_values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let scalar_field = ScalarField1D::new_scalar_field(
            &grid,
            &field_values,
            &BoundaryConditions1D::None,
        )
        .unwrap();
        let result = scalar_field
            .scale(2.0, &BoundaryConditions1D::None)
            .unwrap();
        assert_eq!(result.field_values, vec![2.0, 4.0, 6.0, 8.0, 10.0]);
    }

    #[test]
    fn test_partial_x() {
        let grid = Grid::new_uniform_grid(0.0, 1.0, 100);

        // Test the derivative of a constant scalar field with None boundary
        // conditions.
        let scalar_field = ScalarField1D::new_constant_scalar_field(
            &grid,
            1.0,
            &BoundaryConditions1D::None,
        )
        .unwrap();
        let partial_x_scalar_field = scalar_field.partial_x().unwrap();

        let expected_result = ScalarField1D::new_constant_scalar_field(
            &grid,
            0.0,
            &BoundaryConditions1D::None,
        )
        .unwrap();

        assert!(utils::scalar_field_equality(
            &partial_x_scalar_field,
            &expected_result,
            1e-6
        ));

        // Test the derivative of the scalar field x with None BCs.
        let scalar_field = ScalarField1D::function_to_scalar_field(
            &grid,
            |x| x,
            &BoundaryConditions1D::None,
        )
        .unwrap();
        let partial_x_scalar_field = scalar_field.partial_x().unwrap();

        let expected_result = ScalarField1D::new_constant_scalar_field(
            &grid,
            1.0,
            &BoundaryConditions1D::None,
        )
        .unwrap();

        assert!(utils::scalar_field_equality(
            &partial_x_scalar_field,
            &expected_result,
            1e-6
        ));

        // Test the derivative of the scalar field x^2.
        let scalar_field = ScalarField1D::function_to_scalar_field(
            &grid,
            |x| x.powi(2),
            &BoundaryConditions1D::None,
        )
        .unwrap();
        let partial_x_scalar_field = scalar_field.partial_x().unwrap();

        let expected_result = ScalarField1D::function_to_scalar_field(
            &grid,
            |x| 2.0 * x,
            &BoundaryConditions1D::None,
        )
        .unwrap();

        assert!(utils::scalar_field_equality(
            &partial_x_scalar_field,
            &expected_result,
            1e-6
        ));
    }

    // #[test]
    // fn test_central_difference_derivative_quadratic_at_boundaries() {
    //     let grid = Grid::new_uniform_grid(0.0, 1.0, 100);

    //     // Test the derivative of a constant scalar field.
    //     let scalar_field = ScalarField1D::new_constant_scalar_field(&grid, 1.0);
    //     let partial_x_scalar_field = scalar_field
    //         .central_difference_derivative_quadratic_at_boundaries();

    //     let expected_result =
    //         ScalarField1D::new_constant_scalar_field(&grid, 0.0);

    //     assert!(partial_x_scalar_field.test_equality(&expected_result, 1e-6));

    //     // Test the derivative of the scalar field x.
    //     let scalar_field =
    //         ScalarField1D::function_to_scalar_field(&grid, |x| x);
    //     let partial_x_scalar_field = scalar_field
    //         .central_difference_derivative_quadratic_at_boundaries();

    //     let expected_result =
    //         ScalarField1D::new_constant_scalar_field(&grid, 1.0);

    //     assert!(partial_x_scalar_field.test_equality(&expected_result, 1e-6));

    //     // Test the derivative of the scalar field x^2.
    //     let scalar_field =
    //         ScalarField1D::function_to_scalar_field(&grid, |x| x.powi(2));
    //     let partial_x_scalar_field = scalar_field
    //         .central_difference_derivative_quadratic_at_boundaries();

    //     let expected_result =
    //         ScalarField1D::function_to_scalar_field(&grid, |x| 2.0 * x);

    //     assert!(partial_x_scalar_field.test_equality(&expected_result, 1e-6));
    // }

    #[test]
    fn test_new_scalar_field_error() {
        let grid = Grid::new_uniform_grid(0.0, 1.0, 5);
        let field_values = vec![1.0, 2.0, 3.0];
        let result = ScalarField1D::new_scalar_field(
            &grid,
            &field_values,
            &BoundaryConditions1D::None,
        );
        assert!(result.is_err());
        assert_eq!(
            result.err().unwrap(),
            "Number of field values does not match number of grid points"
        );
    }

    #[test]
    fn test_add_scalar_fields_with_mismatched_grids() {
        let grid1 = Grid::new_uniform_grid(0.0, 1.0, 5);
        let grid2 = Grid::new_uniform_grid(0.0, 2.0, 5);
        let field_values1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let field_values2 = vec![4.0, 3.0, 2.0, 1.0, 0.0];
        let scalar_field1 = ScalarField1D::new_scalar_field(
            &grid1,
            &field_values1,
            &BoundaryConditions1D::None,
        )
        .unwrap();
        let scalar_field2 = ScalarField1D::new_scalar_field(
            &grid2,
            &field_values2,
            &BoundaryConditions1D::None,
        )
        .unwrap();
        let result =
            scalar_field1.add(&scalar_field2, &BoundaryConditions1D::None);
        assert!(result.is_err());
        assert_eq!(result.err().unwrap(), "Grids do not match");
    }

    #[test]
    fn test_subtract_scalar_fields_with_mismatched_grids() {
        let grid1 = Grid::new_uniform_grid(0.0, 1.0, 5);
        let grid2 = Grid::new_uniform_grid(0.0, 2.0, 5);
        let field_values1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let field_values2 = vec![4.0, 3.0, 2.0, 1.0, 0.0];
        let scalar_field1 = ScalarField1D::new_scalar_field(
            &grid1,
            &field_values1,
            &BoundaryConditions1D::None,
        )
        .unwrap();
        let scalar_field2 = ScalarField1D::new_scalar_field(
            &grid2,
            &field_values2,
            &BoundaryConditions1D::None,
        )
        .unwrap();
        let result =
            scalar_field1.subtract(&scalar_field2, &BoundaryConditions1D::None);
        assert!(result.is_err());
        assert_eq!(result.err().unwrap(), "Grids do not match");
    }

    #[test]
    fn test_multiply_scalar_fields_with_mismatched_grids() {
        let grid1 = Grid::new_uniform_grid(0.0, 1.0, 5);
        let grid2 = Grid::new_uniform_grid(0.0, 2.0, 5);
        let field_values1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let field_values2 = vec![4.0, 3.0, 2.0, 1.0, 0.0];
        let scalar_field1 = ScalarField1D::new_scalar_field(
            &grid1,
            &field_values1,
            &BoundaryConditions1D::None,
        )
        .unwrap();
        let scalar_field2 = ScalarField1D::new_scalar_field(
            &grid2,
            &field_values2,
            &BoundaryConditions1D::None,
        )
        .unwrap();
        let result =
            scalar_field1.multiply(&scalar_field2, &BoundaryConditions1D::None);
        assert!(result.is_err());
        assert_eq!(result.err().unwrap(), "Grids do not match");
    }

    #[test]
    fn test_divide_scalar_fields_with_mismatched_grids() {
        let grid1 = Grid::new_uniform_grid(0.0, 1.0, 5);
        let grid2 = Grid::new_uniform_grid(0.0, 2.0, 5);
        let field_values1 = vec![4.0, 9.0, 16.0, 25.0, 36.0];
        let field_values2 = vec![2.0, 3.0, 4.0, 5.0, 0.0];
        let scalar_field1 =
            ScalarField1D::new_scalar_field(&grid1, &field_values1).unwrap();
        let scalar_field2 =
            ScalarField1D::new_scalar_field(&grid1, &field_values2).unwrap();
        let result = scalar_field1.divide(&scalar_field2);
        assert!(result.is_err());
        assert_eq!(result.err().unwrap(), "Division by zero");

        let scalar_field2 = ScalarField1D::new_scalar_field(
            &grid2,
            &vec![2.0, 3.0, 4.0, 5.0, 6.0],
        )
        .unwrap();
        let result = scalar_field1.divide(&scalar_field2);
        assert!(result.is_err());
        assert_eq!(result.err().unwrap(), "Grids do not match");
    }
}
