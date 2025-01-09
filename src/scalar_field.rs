use crate::grid::Grid;

/// # Scalar field 1D
///
/// ## Description
/// A `ScalarField1D` represents a scalar field sampled on a 1D grid of points.
/// Each grid point has an associated scalar, stored as an `f64`. The grid
/// points are represented by a `Grid` instance.
///
/// The x axis is the axis along which the grid points lie.
///
/// ## Example use case
/// ```
/// use crate::grid::Grid;
/// use crate::scalar_field::ScalarField1D;
///
/// let grid = Grid::new_uniform_grid(0.0, 1.0, 5);
/// let field_values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let scalar_field = ScalarField1D::new_scalar_field(&grid, field_values).unwrap();
/// ```
///
#[derive(Debug, Clone, PartialEq)]
pub struct ScalarField1D {
    pub grid: Grid,
    pub field_values: Vec<f64>,
}

// Constructor functions.
impl ScalarField1D {
    /// # New scalar field
    ///
    /// ## Description
    /// Creates a new 1D scalar field, with a specified scalar at each grid
    /// point.
    ///
    /// ## Example use case
    /// ```
    /// use crate::grid::Grid;
    /// use crate::scalar_field::ScalarField1D;
    ///
    /// let grid = Grid::new_uniform_grid(0.0, 1.0, 5);
    /// let field_values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    /// let scalar_field = ScalarField1D::new_scalar_field(&grid, &field_values).unwrap();
    /// ```
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
    /// ## Example use case
    /// ```
    /// use crate::grid::Grid;
    /// use crate::scalar_field::ScalarField1D;
    ///
    /// let grid = Grid::new_uniform_grid(0.0, 1.0, 5);
    /// let scalar_field = ScalarField1D::function_to_scalar_field(&grid, |x| x.powi(2));
    /// ```    
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
        }
    }

    /// # New constant scalar field
    ///
    /// ## Description
    /// Creates a new 1D scalar field with a constant value at every grid point.
    ///
    /// ## Example use case
    /// ```
    /// use crate::grid::Grid;
    /// use crate::scalar_field::ScalarField1D;
    ///
    /// let grid = Grid::new_uniform_grid(0.0, 1.0, 5);
    /// let field_value = 5.0;
    /// let scalar_field = ScalarField1D::new_constant_scalar_field(&grid, field_value);
    /// ```
    ///
    pub fn new_constant_scalar_field(
        grid: &Grid,
        field_value: f64,
    ) -> ScalarField1D {
        let field_values = vec![field_value; grid.grid_points.len()];
        ScalarField1D {
            grid: grid.clone(),
            field_values,
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
    /// ## Example use case
    /// ```
    /// use crate::grid::Grid;
    /// use crate::scalar_field::ScalarField1D;
    ///
    /// let grid = Grid::new_uniform_grid(0.0, 1.0, 5);
    /// let field_values_1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    /// let field_values_2 = vec![4.0, 3.0, 2.0, 1.0, 0.0];
    /// let scalar_field_1 = ScalarField1D::new_scalar_field(&grid, &field_values_1).unwrap();
    /// let scalar_field_2 = ScalarField1D::new_scalar_field(&grid, &field_values_2).unwrap();
    /// let scalar_field_sum = scalar_field1.add(&scalar_field2).unwrap();
    /// ```
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
        })
    }

    /// # Subtract scalar fields
    ///
    /// ## Description
    /// Subtracts `scalar_field` from the current scalar field, and returns this
    /// new scalar field.
    ///
    /// ## Example use case
    /// ```
    /// use crate::grid::Grid;
    /// use crate::scalar_field::ScalarField1D;
    ///
    /// let grid = Grid::new_uniform_grid(0.0, 1.0, 5);
    /// let field_values_1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    /// let field_values_2 = vec![4.0, 3.0, 2.0, 1.0, 0.0];
    /// let scalar_field_1 = ScalarField1D::new_scalar_field(&grid, &field_values_1).unwrap();
    /// let scalar_field_2 = ScalarField1D::new_scalar_field(&grid, &field_values_2).unwrap();
    /// let scalar_field_difference = scalar_field_1.subtract(&scalar_field_2).unwrap();
    /// ```
    ///
    /// ## Todo
    /// Improve error handling.
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
        })
    }

    /// # Multiply scalar fields
    ///
    /// ## Description
    /// Multiplies the current scalar field by `scalar_field`, and returns this
    /// new scalar field.
    ///
    /// ## Example use case
    /// ```
    /// use crate::grid::Grid;
    /// use crate::scalar_field::ScalarField1D;
    ///
    /// let grid = Grid::new_uniform_grid(0.0, 1.0, 5);
    /// let field_values_1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    /// let field_values_2 = vec![4.0, 3.0, 2.0, 1.0, 0.0];
    /// let scalar_field_1 = ScalarField1D::new_scalar_field(&grid, &field_values_1).unwrap();
    /// let scalar_field_2 = ScalarField1D::new_scalar_field(&grid, &field_values_2).unwrap();
    /// let scalar_field_product = scalar_field_1.multiply(&scalar_field_2).unwrap();
    /// ```
    ///
    /// ## Todo
    /// Improve error handling.
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
        })
    }

    /// # Divide scalar fields
    ///
    /// ## Description
    /// Divides the current scalar field by `scalar_field`, and returns this new
    /// scalar field.
    ///
    /// ## Example use case
    /// ```
    /// use crate::grid::Grid;
    /// use crate::scalar_field::ScalarField1D;
    ///
    /// let grid = Grid::new_uniform_grid(0.0, 1.0, 5);
    /// let field_values_1 = vec![4.0, 9.0, 16.0, 25.0, 36.0];
    /// let field_values_2 = vec![2.0, 3.0, 4.0, 5.0, 6.0];
    /// let scalar_field_1 = ScalarField1D::new_scalar_field(&grid, &field_values_1).unwrap();
    /// let scalar_field_2 = ScalarField1D::new_scalar_field(&grid, &field_values_2).unwrap();
    /// let scalar_field_ratio = scalar_field_1.divide(&scalar_field_2).unwrap();
    /// ```
    ///
    /// ## Todo
    /// Improve error handling.
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
        })
    }

    /// # Scale scalar field.
    ///
    /// ## Description
    /// Multiplies the current scalar field by a scalar, and returns this new
    /// scalar field.
    ///
    /// ## Example use case
    /// ```
    /// use crate::grid::Grid;
    /// use crate::scalar_field::ScalarField1D;
    ///
    /// let grid = Grid::new_uniform_grid(0.0, 1.0, 5);
    /// let field_values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    /// let scalar_field = ScalarField1D::new_scalar_field(&grid, &field_values).unwrap();
    /// let scaled_scalar_field = scalar_field.scale(2.0);
    /// ```
    ///
    pub fn scale(&self, scalar: f64) -> Self {
        let field_values: Vec<f64> =
            self.field_values.iter().map(|v| v * scalar).collect();
        ScalarField1D {
            grid: self.grid.clone(),
            field_values,
        }
    }

    /// # Test equality
    ///
    /// ## Description
    /// Tests if the current scalar field is equal to another scalar field
    /// within a specified maximum error.
    ///
    /// ## Example use case
    /// ```
    /// use crate::grid::Grid;
    /// use crate::scalar_field::ScalarField1D;
    ///
    /// let grid = Grid::new_uniform_grid(0.0, 1.0, 5);
    /// let field_values_1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    /// let scalar_field_1 = ScalarField1D::new_scalar_field(&grid, &field_values_1).unwrap();
    /// let scalar_field_2 = scalar_field_1.add(&ScalarField1D::new_constant_scalar_field(&grid, 1e-7)).unwrap();
    /// let is_equal = scalar_field_1.test_equality(&scalar_field_2, 1e-6);
    /// ```
    ///
    pub fn test_equality(
        &self,
        scalar_field: &ScalarField1D,
        max_err: f64,
    ) -> bool {
        if self.grid != scalar_field.grid {
            return false;
        }

        self.field_values
            .iter()
            .zip(&scalar_field.field_values)
            .all(|(v1, v2)| (v1 - v2).abs() < max_err)
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
    /// ## Example use case
    /// ```
    /// use crate::grid::Grid;
    /// use crate::scalar_field::ScalarField1D;
    ///
    /// let grid = Grid::new_uniform_grid(0.0, 1.0, 5);
    /// let field_values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    /// let scalar_field = ScalarField1D::new_scalar_field(&grid, &field_values).unwrap();
    /// let scalar_field_derivative = scalar_field.partial_x();
    /// ```
    ///  
    pub fn partial_x(&self) -> Self {
        self.central_difference_derivative()
    }

    /// # Central difference derivative
    ///
    /// ## Description
    /// Calculates the derivative of the scalar field using the central
    /// difference scheme, and returns this result as a new scalar field.
    ///
    /// The derivative at the starting grid point is calculated using the
    /// forwards difference scheme, the derivative at each interior grid point
    /// is calculated using the central difference scheme, and the derivative
    /// at the final grid point is calculated using the backwards difference
    /// scheme.
    ///
    /// ## Example use case
    /// ```
    /// use crate::grid::Grid;
    /// use crate::scalar_field::ScalarField1D;
    ///
    /// let grid = Grid::new_uniform_grid(0.0, 1.0, 5);
    /// let field_values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    /// let scalar_field = ScalarField1D::new_scalar_field(&grid, &field_values).unwrap();
    /// let scalar_field_derivative = scalar_field.central_difference_derivative();
    /// ```
    ///
    pub fn central_difference_derivative(self: &Self) -> Self {
        let grid = &self.grid;
        let field_values = &self.field_values;

        let grid_points = &grid.grid_points;
        let num_points = grid_points.len();

        let mut partial_x_values = Vec::new();

        // Calculates the derivative at the starting grid point using the forwards difference scheme.
        partial_x_values.push(
            (field_values[1] - field_values[0])
                / (grid_points[1] - grid_points[0]),
        );

        // Calculates the derivative at each interior grid point using the central difference scheme.
        for i in 1..(num_points - 1) {
            partial_x_values.push(
                (field_values[i + 1] - field_values[i - 1])
                    / (grid_points[i + 1] - grid_points[i - 1]),
            );
        }

        // Calculates the derivative at the final grid point using the backwards difference scheme.
        partial_x_values.push(
            (field_values[num_points - 1] - field_values[num_points - 2])
                / (grid_points[num_points - 1] - grid_points[num_points - 2]),
        );

        ScalarField1D {
            grid: grid.clone(),
            field_values: partial_x_values,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::Grid;

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

        assert!(scalar_field.test_equality(&expected_scalar_field, 1e-6));
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
    fn test_test_equality() {
        let grid = Grid::new_uniform_grid(0.0, 1.0, 5);
        let field_values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let scalar_field_1 =
            ScalarField1D::new_scalar_field(&grid, &field_values).unwrap();

        // Test equality of a scalar field with itself.
        assert!(scalar_field_1.test_equality(&scalar_field_1, 1e-6));

        // Test equality of a scalar field with a slightly different scalar
        // field.
        let mut scalar_field_2 = scalar_field_1
            .add(&ScalarField1D::new_constant_scalar_field(&grid, 1e-7))
            .unwrap();
        assert!(scalar_field_1.test_equality(&scalar_field_2, 1e-6));

        // Test inequality of a scalar field with a different scalar field.
        scalar_field_2 = scalar_field_1
            .add(&ScalarField1D::new_constant_scalar_field(&grid, 1e-3))
            .unwrap();
        assert!(!scalar_field_1.test_equality(&scalar_field_2, 1e-6));
    }

    #[test]
    fn test_partial_x() {
        let grid = Grid::new_uniform_grid(0.0, 1.0, 100);

        // Test the derivative of a constant scalar field.
        let scalar_field = ScalarField1D::new_constant_scalar_field(&grid, 1.0);
        let partial_x_scalar_field = scalar_field.partial_x();

        let expected_result =
            ScalarField1D::new_constant_scalar_field(&grid, 0.0);

        assert!(partial_x_scalar_field.test_equality(&expected_result, 1e-6));

        // Test the derivative of the scalar field x.
        let scalar_field =
            ScalarField1D::function_to_scalar_field(&grid, |x| x);
        let partial_x_scalar_field = scalar_field.partial_x();

        let expected_result =
            ScalarField1D::new_constant_scalar_field(&grid, 1.0);

        assert!(partial_x_scalar_field.test_equality(&expected_result, 1e-6));
    }

    #[test]
    fn test_central_difference_derivative() {
        let grid = Grid::new_uniform_grid(0.0, 1.0, 100);

        // Test the derivative of a constant scalar field.
        let scalar_field = ScalarField1D::new_constant_scalar_field(&grid, 1.0);
        let partial_x_scalar_field =
            scalar_field.central_difference_derivative();

        let expected_result =
            ScalarField1D::new_constant_scalar_field(&grid, 0.0);

        assert!(partial_x_scalar_field.test_equality(&expected_result, 1e-6));

        // Test the derivative of the scalar field x.
        let scalar_field =
            ScalarField1D::function_to_scalar_field(&grid, |x| x);
        let partial_x_scalar_field =
            scalar_field.central_difference_derivative();

        let expected_result =
            ScalarField1D::new_constant_scalar_field(&grid, 1.0);

        assert!(partial_x_scalar_field.test_equality(&expected_result, 1e-6));
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
