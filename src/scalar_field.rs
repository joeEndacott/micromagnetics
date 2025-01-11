use crate::boundary_conditions;
use crate::boundary_conditions::BoundaryConditions1D;
use crate::grid::Grid;
use crate::quadratic_interpolation;
use crate::utils;

/// # Scalar field 1D
///
/// ## Description
/// Represents a scalar field sampled on a 1D grid of points. Each grid point
/// has an associated scalar, stored as an `f64`. The grid points are
/// represented by a `Grid` instance.
///
/// The x axis is the axis along which the grid points lie.
///
/// The scalar field includes details about the boundary conditions
/// applied to it.
///
#[derive(Debug, Clone, PartialEq)]
pub struct ScalarField1D {
    pub grid: Grid,
    pub field_values: Vec<f64>,
    boundary_conditions: BoundaryConditions1D,
}

// Constructor functions.
impl ScalarField1D {
    /// # New scalar field
    ///
    /// ## Description
    /// Creates a new 1D scalar field, with a specified scalar at each grid
    /// point.
    ///
    /// The scalar field is given the default boundary conditions.
    ///
    pub fn new_scalar_field(
        grid: &Grid,
        field_values: &Vec<f64>,
    ) -> Result<ScalarField1D, &'static str> {
        if field_values.len() != grid.grid_points.len() {
            return Err(
                "Number of field values does not match number of grid points",
            );
        }
        Ok(ScalarField1D {
            grid: grid.clone(),
            field_values: field_values.clone(),
            boundary_conditions: BoundaryConditions1D::default(),
        })
    }

    /// # Function to scalar field
    ///
    /// ## Description
    /// Generates a `ScalarField1D`, given a real-valued function of a real
    /// variable `func` and a `Grid` `grid`. `func` is
    /// sampled at each grid point in `grid` and the values are stored
    /// in the field `field_values`.
    ///
    /// The scalar field is given the default boundary conditions.
    ///
    pub fn function_to_scalar_field<F>(grid: &Grid, func: F) -> Self
    where
        F: Fn(f64) -> f64,
    {
        // Creates a vector containing the value of func at each grid point.
        let function_values: Vec<f64> =
            grid.grid_points.iter().map(|&x| func(x)).collect();

        ScalarField1D {
            grid: grid.clone(),
            field_values: function_values,
            boundary_conditions: BoundaryConditions1D::default(),
        }
    }

    /// # New constant scalar field
    ///
    /// ## Description
    /// Creates a new 1D scalar field with a constant value at every grid point.
    ///
    /// The scalar field is given the default boundary conditions.
    ///
    pub fn new_constant_scalar_field(
        grid: &Grid,
        field_value: f64,
    ) -> ScalarField1D {
        let field_values = vec![field_value; grid.grid_points.len()];
        ScalarField1D {
            grid: grid.clone(),
            field_values,
            boundary_conditions: BoundaryConditions1D::default(),
        }
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
    /// The new scalar field is given the default boundary conditions.
    ///
    /// ## Todo
    /// Improve error handling.
    ///
    pub fn add(
        &self,
        scalar_field: &ScalarField1D,
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

        Ok(ScalarField1D {
            grid: self.grid.clone(),
            field_values,
            boundary_conditions: BoundaryConditions1D::default(),
        })
    }

    /// # Subtract scalar fields
    ///
    /// ## Description
    /// Subtracts `scalar_field` from the current scalar field, and returns this
    /// new scalar field.
    ///
    /// The new scalar field is given the default boundary conditions.
    ///
    pub fn subtract(
        &self,
        scalar_field: &ScalarField1D,
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

        Ok(ScalarField1D {
            grid: self.grid.clone(),
            field_values,
            boundary_conditions: BoundaryConditions1D::default(),
        })
    }

    /// # Multiply scalar fields
    ///
    /// ## Description
    /// Multiplies the current scalar field by `scalar_field`, and returns this
    /// new scalar field.
    ///
    /// The new scalar field is given the default boundary conditions.
    ///
    pub fn multiply(
        &self,
        scalar_field: &ScalarField1D,
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

        Ok(ScalarField1D {
            grid: self.grid.clone(),
            field_values,
            boundary_conditions: BoundaryConditions1D::default(),
        })
    }

    /// # Divide scalar fields
    ///
    /// ## Description
    /// Divides the current scalar field by `scalar_field`, and returns this new
    /// scalar field.
    ///
    /// The new scalar field is given the default boundary conditions.
    ///
    pub fn divide(
        &self,
        scalar_field: &ScalarField1D,
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

        Ok(ScalarField1D {
            grid: self.grid.clone(),
            field_values,
            boundary_conditions: BoundaryConditions1D::default(),
        })
    }

    /// # Scale scalar field.
    ///
    /// ## Description
    /// Multiplies the current scalar field by a scalar, and returns this new
    /// scalar field.
    ///
    /// The new scalar field is given the default boundary conditions.
    ///
    pub fn scale(&self, scalar: f64) -> Self {
        let field_values: Vec<f64> =
            self.field_values.iter().map(|v| v * scalar).collect();
        ScalarField1D {
            grid: self.grid.clone(),
            field_values,
            boundary_conditions: BoundaryConditions1D::default(),
        }
    }
}

// Boundary conditions.
impl ScalarField1D {
    /// # Check boundary conditions are satisfied for a scalar field
    ///
    /// ## Description
    /// Checks that the specified boundary conditions are satisfied for a
    /// scalar field within a set tolerance.
    ///
    /// If the boundary conditions are satisfied, returns `Ok(())`, else returns
    /// an error message.
    ///
    /// The specified boundary conditions do not need to be the same as the
    /// boundary conditions of the scalar field. For example, the scalar field
    /// may have None boundary conditions, but the user may want to check
    /// if the field satisfies Dirichlet boundary conditions, so that the
    /// boundary conditions of the field can be changed.
    ///
    pub fn check_scalar_bcs(
        self: &Self,
        boundary_conditions: &BoundaryConditions1D,
        tolerance: f64,
    ) -> Result<(), &'static str> {
        let num_points = self.field_values.len();
        if num_points < 2 {
            return Err("At least two grid points are required");
        }

        match boundary_conditions {
            BoundaryConditions1D::None => Ok(()),
            BoundaryConditions1D::Periodic => {
                if !utils::scalar_equality(
                    self.field_values[0],
                    self.field_values[num_points - 1],
                    tolerance,
                ) {
                    return Err("Periodic BCs: Field values do not match at the boundaries");
                }
                Ok(())
            }
            BoundaryConditions1D::DirichletScalar(
                left_boundary_value,
                right_boundary_value,
            ) => {
                if !utils::scalar_equality(
                    self.field_values[0],
                    *left_boundary_value,
                    tolerance,
                ) || !utils::scalar_equality(
                    self.field_values[num_points - 1],
                    *right_boundary_value,
                    tolerance,
                ) {
                    return Err("Dirichlet BCs: Field values do not match specified boundary values");
                }
                Ok(())
            }
            BoundaryConditions1D::NeumannScalar(_, _) => {
                // Implement Neumann BC logic.
                Ok(())
            }
            BoundaryConditions1D::DirichletVector(_, _)
            | BoundaryConditions1D::NeumannVector(_, _) => {
                return Err("Vector BCs are not supported for scalar fields");
            }
        }
    }

    /// # Apply scalar boundary conditions
    ///
    /// ## Description
    /// Applies the specified boundary conditions to the scalar field.
    ///
    /// If the boundary conditions are successfully applied, returns `Ok(())`,
    /// else returns an error message.
    ///
    /// The specified boundary conditions must be compatible with the scalar
    /// field's values.
    ///
    pub fn apply_scalar_bcs(
        self: &mut Self,
        boundary_conditions: &BoundaryConditions1D,
    ) -> Result<(), &'static str> {
        const TOLERANCE: f64 = 1e-6;

        self.check_scalar_bcs(boundary_conditions, TOLERANCE)?;

        let num_points = self.field_values.len();

        match boundary_conditions {
            BoundaryConditions1D::None => {
                self.boundary_conditions = BoundaryConditions1D::None;
                Ok(())
            }
            BoundaryConditions1D::Periodic => {
                self.boundary_conditions = BoundaryConditions1D::Periodic;
                self.field_values[0] = self.field_values[num_points - 1];
                Ok(())
            }
            BoundaryConditions1D::DirichletScalar(
                left_boundary_value,
                right_boundary_value,
            ) => {
                self.boundary_conditions =
                    BoundaryConditions1D::DirichletScalar(
                        *left_boundary_value,
                        *right_boundary_value,
                    );
                self.field_values[0] = *left_boundary_value;
                self.field_values[num_points - 1] = *right_boundary_value;
                Ok(())
            }
            BoundaryConditions1D::NeumannScalar(
                left_boundary_value,
                right_boundary_value,
            ) => {
                self.boundary_conditions = BoundaryConditions1D::NeumannScalar(
                    *left_boundary_value,
                    *right_boundary_value,
                );
                // Implement Neumann BC logic.
                Ok(())
            }
            BoundaryConditions1D::DirichletVector(_, _)
            | BoundaryConditions1D::NeumannVector(_, _) => {
                return Err("Vector BCs are not supported for scalar fields");
            }
        }
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
    /// Different differentiation schemes are used for different boundary
    /// conditions.
    ///  
    pub fn partial_x(&self) -> Self {
        self.central_difference_derivative_quadratic_at_boundaries()
    }

    /// # Central difference derivative
    ///
    /// ## Description
    /// Calculates the partial derivative of the scalar field with respect to x
    /// using the central difference scheme, and returns this result as a new
    /// scalar field.
    ///
    /// - The derivative at the starting grid point is calculated using a     
    /// quadratic interpolation.
    /// - The derivative at each interior grid point is calculated using the
    /// central difference scheme.
    /// - The derivative at the final grid point is calculated using a quadratic
    /// interpolation.
    ///
    /// Currently, the resulting scalar field is given the default boundary
    /// conditions.
    ///
    pub fn central_difference_derivative_quadratic_at_boundaries(
        self: &Self,
    ) -> Self {
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
            boundary_conditions: BoundaryConditions1D::default(),
        }
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
        let scalar_field =
            ScalarField1D::new_scalar_field(&grid, &field_values).unwrap();
        assert_eq!(scalar_field.grid, grid);
        assert_eq!(scalar_field.field_values, field_values);
    }

    #[test]
    fn test_function_to_scalar_field() {
        let grid = Grid::new_uniform_grid(0.0, 1.0, 6);
        let scalar_field =
            ScalarField1D::function_to_scalar_field(&grid, |x| x);

        let expected_values = vec![0.0, 0.2, 0.4, 0.6, 0.8, 1.0];
        let expected_scalar_field =
            ScalarField1D::new_scalar_field(&grid, &expected_values).unwrap();

        assert!(utils::scalar_field_equality(
            &scalar_field,
            &expected_scalar_field,
            1e-6
        ));
    }

    #[test]
    fn test_new_constant_scalar_field() {
        let grid = Grid::new_uniform_grid(0.0, 1.0, 5);
        let field_value = 5.0;
        let scalar_field =
            ScalarField1D::new_constant_scalar_field(&grid, field_value);
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
        let scalar_field1 =
            ScalarField1D::new_scalar_field(&grid, &field_values1).unwrap();
        let scalar_field2 =
            ScalarField1D::new_scalar_field(&grid, &field_values2).unwrap();
        let result = scalar_field1.add(&scalar_field2).unwrap();
        assert_eq!(result.field_values, vec![5.0, 5.0, 5.0, 5.0, 5.0]);
    }

    #[test]
    fn test_subtract() {
        let grid = Grid::new_uniform_grid(0.0, 1.0, 5);
        let field_values1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let field_values2 = vec![4.0, 3.0, 2.0, 1.0, 0.0];
        let scalar_field1 =
            ScalarField1D::new_scalar_field(&grid, &field_values1).unwrap();
        let scalar_field2 =
            ScalarField1D::new_scalar_field(&grid, &field_values2).unwrap();
        let result = scalar_field1.subtract(&scalar_field2).unwrap();
        assert_eq!(result.field_values, vec![-3.0, -1.0, 1.0, 3.0, 5.0]);
    }

    #[test]
    fn test_multiply() {
        let grid = Grid::new_uniform_grid(0.0, 1.0, 5);
        let field_values1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let field_values2 = vec![4.0, 3.0, 2.0, 1.0, 0.0];
        let scalar_field1 =
            ScalarField1D::new_scalar_field(&grid, &field_values1).unwrap();
        let scalar_field2 =
            ScalarField1D::new_scalar_field(&grid, &field_values2).unwrap();
        let result = scalar_field1.multiply(&scalar_field2).unwrap();
        assert_eq!(result.field_values, vec![4.0, 6.0, 6.0, 4.0, 0.0]);
    }

    #[test]
    fn test_divide() {
        let grid = Grid::new_uniform_grid(0.0, 1.0, 5);
        let field_values1 = vec![4.0, 9.0, 16.0, 25.0, 36.0];
        let field_values2 = vec![2.0, 3.0, 4.0, 5.0, 6.0];
        let scalar_field1 =
            ScalarField1D::new_scalar_field(&grid, &field_values1).unwrap();
        let scalar_field2 =
            ScalarField1D::new_scalar_field(&grid, &field_values2).unwrap();
        let result = scalar_field1.divide(&scalar_field2).unwrap();
        assert_eq!(result.field_values, vec![2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_scale() {
        let grid = Grid::new_uniform_grid(0.0, 1.0, 5);
        let field_values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let scalar_field =
            ScalarField1D::new_scalar_field(&grid, &field_values).unwrap();
        let result = scalar_field.scale(2.0);
        assert_eq!(result.field_values, vec![2.0, 4.0, 6.0, 8.0, 10.0]);
    }

    #[test]
    fn test_partial_x() {
        let grid = Grid::new_uniform_grid(0.0, 1.0, 100);

        // Test the derivative of a constant scalar field.
        let scalar_field = ScalarField1D::new_constant_scalar_field(&grid, 1.0);
        let partial_x_scalar_field = scalar_field.partial_x();

        let expected_result =
            ScalarField1D::new_constant_scalar_field(&grid, 0.0);

        assert!(utils::scalar_field_equality(
            &partial_x_scalar_field,
            &expected_result,
            1e-6
        ));

        // Test the derivative of the scalar field x.
        let scalar_field =
            ScalarField1D::function_to_scalar_field(&grid, |x| x);
        let partial_x_scalar_field = scalar_field.partial_x();

        let expected_result =
            ScalarField1D::new_constant_scalar_field(&grid, 1.0);

        assert!(utils::scalar_field_equality(
            &partial_x_scalar_field,
            &expected_result,
            1e-6
        ));

        // Test the derivative of the scalar field x^2.
        let scalar_field =
            ScalarField1D::function_to_scalar_field(&grid, |x| x.powi(2));
        let partial_x_scalar_field = scalar_field.partial_x();

        let expected_result =
            ScalarField1D::function_to_scalar_field(&grid, |x| 2.0 * x);

        assert!(utils::scalar_field_equality(
            &partial_x_scalar_field,
            &expected_result,
            1e-6
        ));
    }

    #[test]
    fn test_central_difference_derivative_quadratic_at_boundaries() {
        let grid = Grid::new_uniform_grid(0.0, 1.0, 100);

        // Test the derivative of a constant scalar field.
        let scalar_field = ScalarField1D::new_constant_scalar_field(&grid, 1.0);
        let partial_x_scalar_field = scalar_field
            .central_difference_derivative_quadratic_at_boundaries();

        let expected_result =
            ScalarField1D::new_constant_scalar_field(&grid, 0.0);

        assert!(utils::scalar_field_equality(
            &partial_x_scalar_field,
            &expected_result,
            1e-6
        ));

        // Test the derivative of the scalar field x.
        let scalar_field =
            ScalarField1D::function_to_scalar_field(&grid, |x| x);
        let partial_x_scalar_field = scalar_field
            .central_difference_derivative_quadratic_at_boundaries();

        let expected_result =
            ScalarField1D::new_constant_scalar_field(&grid, 1.0);

        assert!(utils::scalar_field_equality(
            &partial_x_scalar_field,
            &expected_result,
            1e-6
        ));

        // Test the derivative of the scalar field x^2.
        let scalar_field =
            ScalarField1D::function_to_scalar_field(&grid, |x| x.powi(2));
        let partial_x_scalar_field = scalar_field
            .central_difference_derivative_quadratic_at_boundaries();

        let expected_result =
            ScalarField1D::function_to_scalar_field(&grid, |x| 2.0 * x);

        assert!(utils::scalar_field_equality(
            &partial_x_scalar_field,
            &expected_result,
            1e-6
        ));
    }

    #[test]
    fn test_new_scalar_field_error() {
        let grid = Grid::new_uniform_grid(0.0, 1.0, 5);
        let field_values = vec![1.0, 2.0, 3.0];
        let result = ScalarField1D::new_scalar_field(&grid, &field_values);
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
        let scalar_field1 =
            ScalarField1D::new_scalar_field(&grid1, &field_values1).unwrap();
        let scalar_field2 =
            ScalarField1D::new_scalar_field(&grid2, &field_values2).unwrap();
        let result = scalar_field1.add(&scalar_field2);
        assert!(result.is_err());
        assert_eq!(result.err().unwrap(), "Grids do not match");
    }

    #[test]
    fn test_subtract_scalar_fields_with_mismatched_grids() {
        let grid1 = Grid::new_uniform_grid(0.0, 1.0, 5);
        let grid2 = Grid::new_uniform_grid(0.0, 2.0, 5);
        let field_values1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let field_values2 = vec![4.0, 3.0, 2.0, 1.0, 0.0];
        let scalar_field1 =
            ScalarField1D::new_scalar_field(&grid1, &field_values1).unwrap();
        let scalar_field2 =
            ScalarField1D::new_scalar_field(&grid2, &field_values2).unwrap();
        let result = scalar_field1.subtract(&scalar_field2);
        assert!(result.is_err());
        assert_eq!(result.err().unwrap(), "Grids do not match");
    }

    #[test]
    fn test_multiply_scalar_fields_with_mismatched_grids() {
        let grid1 = Grid::new_uniform_grid(0.0, 1.0, 5);
        let grid2 = Grid::new_uniform_grid(0.0, 2.0, 5);
        let field_values1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let field_values2 = vec![4.0, 3.0, 2.0, 1.0, 0.0];
        let scalar_field1 =
            ScalarField1D::new_scalar_field(&grid1, &field_values1).unwrap();
        let scalar_field2 =
            ScalarField1D::new_scalar_field(&grid2, &field_values2).unwrap();
        let result = scalar_field1.multiply(&scalar_field2);
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
