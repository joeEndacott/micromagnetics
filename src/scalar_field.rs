use crate::grid::Grid;

/// # Scalar field 1D
///
/// ## Description
/// A `ScalarField1D` represents a scalar field sampled on a 1D grid of points.
/// Each grid point has an associated scalar, stored as an `f64`. The grid
/// points are represented by a `Grid` instance.
///
/// ## Example use case
/// ```
/// use crate::grid::Grid;
/// use crate::scalar_field::ScalarField1D;
///
/// let grid = Grid::new_uniform_grid(0.0, 1.0, 5);
/// let field_values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let scalar_field = ScalarField1D::new_scalar_field(grid, field_values).unwrap();
/// println!("{:?}", scalar_field);
/// ```
///
#[derive(Debug, Clone, PartialEq)]
pub struct ScalarField1D {
    pub grid: Grid,
    pub field_values: Vec<f64>,
}

// Constructor functions.
impl ScalarField1D {
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
    /// let scalar_field = ScalarField1D::new_scalar_field(grid, field_values).unwrap();
    /// println!("{:?}", scalar_field);
    /// ```
    ///
    pub fn new_scalar_field(
        grid: Grid,
        field_values: Vec<f64>,
    ) -> Result<ScalarField1D, &'static str> {
        if field_values.len() != grid.grid_points.len() {
            return Err(
                "Number of field values does not match number of grid points",
            );
        }
        Ok(ScalarField1D { grid, field_values })
    }

    /// Creates a new 1D scalar field with a constant value at every grid point.
    ///
    /// ## Example use case
    /// ```
    /// use crate::grid::Grid;
    /// use crate::scalar_field::ScalarField1D;
    ///
    /// let grid = Grid::new_uniform_grid(0.0, 1.0, 5);
    /// let field_value = 5.0;
    /// let scalar_field = ScalarField1D::new_constant_scalar_field(grid, field_value);
    /// println!("{:?}", scalar_field);
    /// ```
    ///
    pub fn new_constant_scalar_field(
        grid: Grid,
        field_value: f64,
    ) -> ScalarField1D {
        let field_values = vec![field_value; grid.grid_points.len()];
        ScalarField1D { grid, field_values }
    }
}

// Arithmetic operations.
impl ScalarField1D {
    /// Adds `scalar_field` to the current scalar field, and returns this new
    /// scalar field.
    ///
    /// ## Example use case
    /// ```
    /// use crate::grid::Grid;
    /// use crate::scalar_field::ScalarField1D;
    ///
    /// let grid = Grid::new_uniform_grid(0.0, 1.0, 5);
    /// let field_values1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    /// let field_values2 = vec![4.0, 3.0, 2.0, 1.0, 0.0];
    /// let scalar_field1 = ScalarField1D::new_scalar_field(grid.clone(), field_values1).unwrap();
    /// let scalar_field2 = ScalarField1D::new_scalar_field(grid, field_values2).unwrap();
    /// let result = scalar_field1.add(&scalar_field2).unwrap();
    /// println!("{:?}", result);
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

    /// Subtracts `scalar_field` from the current scalar field, and returns this
    /// new scalar field.
    ///
    /// ## Example use case
    /// ```
    /// use crate::grid::Grid;
    /// use crate::scalar_field::ScalarField1D;
    ///
    /// let grid = Grid::new_uniform_grid(0.0, 1.0, 5);
    /// let field_values1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    /// let field_values2 = vec![4.0, 3.0, 2.0, 1.0, 0.0];
    /// let scalar_field1 = ScalarField1D::new_scalar_field(grid.clone(), field_values1).unwrap();
    /// let scalar_field2 = ScalarField1D::new_scalar_field(grid, field_values2).unwrap();
    /// let result = scalar_field1.subtract(&scalar_field2).unwrap();
    /// println!("{:?}", result);
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

    /// Multiplies the current scalar field by `scalar_field`, and returns this
    /// new scalar field.
    ///
    /// ## Example use case
    /// ```
    /// use crate::grid::Grid;
    /// use crate::scalar_field::ScalarField1D;
    ///
    /// let grid = Grid::new_uniform_grid(0.0, 1.0, 5);
    /// let field_values1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    /// let field_values2 = vec![4.0, 3.0, 2.0, 1.0, 0.0];
    /// let scalar_field1 = ScalarField1D::new_scalar_field(grid.clone(), field_values1).unwrap();
    /// let scalar_field2 = ScalarField1D::new_scalar_field(grid, field_values2).unwrap();
    /// let result = scalar_field1.multiply(&scalar_field2).unwrap();
    /// println!("{:?}", result);
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

    /// Divides the current scalar field by `scalar_field`, and returns this new
    /// scalar field.
    ///
    /// ## Example use case
    /// ```
    /// use crate::grid::Grid;
    /// use crate::scalar_field::ScalarField1D;
    ///
    /// let grid = Grid::new_uniform_grid(0.0, 1.0, 5);
    /// let field_values1 = vec![4.0, 9.0, 16.0, 25.0, 36.0];
    /// let field_values2 = vec![2.0, 3.0, 4.0, 5.0, 6.0];
    /// let scalar_field1 = ScalarField1D::new_scalar_field(grid.clone(), field_values1).unwrap();
    /// let scalar_field2 = ScalarField1D::new_scalar_field(grid, field_values2).unwrap();
    /// let result = scalar_field1.divide(&scalar_field2).unwrap();
    /// println!("{:?}", result);
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
    /// let scalar_field = ScalarField1D::new_scalar_field(grid, field_values).unwrap();
    /// let result = scalar_field.scale(2.0);
    /// println!("{:?}", result);
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
            ScalarField1D::new_scalar_field(grid.clone(), field_values.clone())
                .unwrap();
        assert_eq!(scalar_field.grid, grid);
        assert_eq!(scalar_field.field_values, field_values);
    }

    #[test]
    fn test_new_constant_scalar_field() {
        let grid = Grid::new_uniform_grid(0.0, 1.0, 5);
        let field_value = 5.0;
        let scalar_field =
            ScalarField1D::new_constant_scalar_field(grid.clone(), field_value);
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
            ScalarField1D::new_scalar_field(grid.clone(), field_values1)
                .unwrap();
        let scalar_field2 =
            ScalarField1D::new_scalar_field(grid.clone(), field_values2)
                .unwrap();
        let result = scalar_field1.add(&scalar_field2).unwrap();
        assert_eq!(result.field_values, vec![5.0, 5.0, 5.0, 5.0, 5.0]);
    }

    #[test]
    fn test_subtract() {
        let grid = Grid::new_uniform_grid(0.0, 1.0, 5);
        let field_values1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let field_values2 = vec![4.0, 3.0, 2.0, 1.0, 0.0];
        let scalar_field1 =
            ScalarField1D::new_scalar_field(grid.clone(), field_values1)
                .unwrap();
        let scalar_field2 =
            ScalarField1D::new_scalar_field(grid.clone(), field_values2)
                .unwrap();
        let result = scalar_field1.subtract(&scalar_field2).unwrap();
        assert_eq!(result.field_values, vec![-3.0, -1.0, 1.0, 3.0, 5.0]);
    }

    #[test]
    fn test_multiply() {
        let grid = Grid::new_uniform_grid(0.0, 1.0, 5);
        let field_values1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let field_values2 = vec![4.0, 3.0, 2.0, 1.0, 0.0];
        let scalar_field1 =
            ScalarField1D::new_scalar_field(grid.clone(), field_values1)
                .unwrap();
        let scalar_field2 =
            ScalarField1D::new_scalar_field(grid.clone(), field_values2)
                .unwrap();
        let result = scalar_field1.multiply(&scalar_field2).unwrap();
        assert_eq!(result.field_values, vec![4.0, 6.0, 6.0, 4.0, 0.0]);
    }

    #[test]
    fn test_divide() {
        let grid = Grid::new_uniform_grid(0.0, 1.0, 5);
        let field_values1 = vec![4.0, 9.0, 16.0, 25.0, 36.0];
        let field_values2 = vec![2.0, 3.0, 4.0, 5.0, 6.0];
        let scalar_field1 =
            ScalarField1D::new_scalar_field(grid.clone(), field_values1)
                .unwrap();
        let scalar_field2 =
            ScalarField1D::new_scalar_field(grid.clone(), field_values2)
                .unwrap();
        let result = scalar_field1.divide(&scalar_field2).unwrap();
        assert_eq!(result.field_values, vec![2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_scale() {
        let grid = Grid::new_uniform_grid(0.0, 1.0, 5);
        let field_values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let scalar_field =
            ScalarField1D::new_scalar_field(grid.clone(), field_values)
                .unwrap();
        let result = scalar_field.scale(2.0);
        assert_eq!(result.field_values, vec![2.0, 4.0, 6.0, 8.0, 10.0]);
    }

    #[test]
    fn test_new_scalar_field_error() {
        let grid = Grid::new_uniform_grid(0.0, 1.0, 5);
        let field_values = vec![1.0, 2.0, 3.0];
        let result = ScalarField1D::new_scalar_field(grid, field_values);
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
            ScalarField1D::new_scalar_field(grid1, field_values1).unwrap();
        let scalar_field2 =
            ScalarField1D::new_scalar_field(grid2, field_values2).unwrap();
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
            ScalarField1D::new_scalar_field(grid1, field_values1).unwrap();
        let scalar_field2 =
            ScalarField1D::new_scalar_field(grid2, field_values2).unwrap();
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
            ScalarField1D::new_scalar_field(grid1, field_values1).unwrap();
        let scalar_field2 =
            ScalarField1D::new_scalar_field(grid2, field_values2).unwrap();
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
            ScalarField1D::new_scalar_field(grid1.clone(), field_values1)
                .unwrap();
        let scalar_field2 =
            ScalarField1D::new_scalar_field(grid1.clone(), field_values2)
                .unwrap();
        let result = scalar_field1.divide(&scalar_field2);
        assert!(result.is_err());
        assert_eq!(result.err().unwrap(), "Division by zero");

        let scalar_field2 = ScalarField1D::new_scalar_field(
            grid2,
            vec![2.0, 3.0, 4.0, 5.0, 6.0],
        )
        .unwrap();
        let result = scalar_field1.divide(&scalar_field2);
        assert!(result.is_err());
        assert_eq!(result.err().unwrap(), "Grids do not match");
    }

    #[test]
    fn test_new_scalar_field_error_handling() {
        let grid = Grid::new_uniform_grid(0.0, 1.0, 5);
        let field_values = vec![1.0, 2.0, 3.0]; // Incorrect length
        let result = ScalarField1D::new_scalar_field(grid, field_values);
        assert!(result.is_err());
        assert_eq!(
            result.err().unwrap(),
            "Number of field values does not match number of grid points"
        );
    }
}
