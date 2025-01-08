use crate::{grid::Grid, scalar_field::ScalarField1D};

/// # Vector field 1D
///
/// ## Description
/// A `VectorField1D` represents a 3-vector field sampled on a 1D grid of
/// points. Each grid point has an associated 3-vector, stored as an array of
/// three `f64` elements. The grid points are represented by a `Grid` instance.
///
/// ## Example use case
/// Todo: add example use case.
///
#[derive(Debug, Clone, PartialEq)]
pub struct VectorField1D {
    pub grid: Grid,
    pub field_values: Vec<[f64; 3]>,
}

// Constructor functions.
impl VectorField1D {
    /// # New vector field
    ///
    /// ## Description
    /// Creates a new 1D 3-vector field, with a specified 3-vector at each grid
    /// point.
    ///
    /// ## Example use case
    /// ```
    /// let grid = Grid::new_uniform_grid(0.0, 1.0, 11);
    /// let field_values = vec![[1.0, 0.0, 0.0]; grid.grid_points.len()];
    /// let vector_field = VectorField1D::new_vector_field(&grid, field_values);
    /// ```
    ///
    pub fn new_vector_field(
        grid: &Grid,
        field_values: Vec<[f64; 3]>,
    ) -> Result<VectorField1D, &'static str> {
        if grid.grid_points.len() != field_values.len() {
            return Err(
                "Number of grid points does not match number of field values",
            );
        }

        let grid = grid.clone();

        Ok(VectorField1D { grid, field_values })
    }

    /// # New constant vector field
    ///
    /// ## Description
    /// Creates a new 1D 3-vector field with a constant 3D vector at every grid
    /// point.
    ///
    /// ## Example use case
    /// ```
    /// let grid = Grid::new_uniform_grid(0.0, 1.0, 11);
    /// let field_value = [1.0, 0.0, 0.0];
    /// let vector_field = VectorField1D::new_constant_vector_field(&grid, field_value);
    /// ```
    ///
    pub fn new_constant_vector_field(
        grid: &Grid,
        field_value: [f64; 3],
    ) -> VectorField1D {
        let grid = grid.clone();
        let field_values = vec![field_value; grid.grid_points.len()];

        VectorField1D { grid, field_values }
    }
}

// Arithmetic operations.
impl VectorField1D {
    /// # Add vector fields
    ///
    /// ## Description
    /// Adds `vector_field` to the current vector field, and returns this new
    /// vector field.
    ///
    /// ## Example use case
    /// ```
    /// let grid = Grid::new_uniform_grid(0.0, 1.0, 11);
    /// let field_values_1 = vec![[1.0, 0.0, 0.0]; grid.grid_points.len()];
    /// let field_values_2 = vec![[0.0, 1.0, 0.0]; grid.grid_points.len()];
    /// let vector_field_1 = VectorField1D::new_vector_field(&grid, field_values_1);
    /// let vector_field_2 = VectorField1D::new_vector_field(&grid, field_values_2);
    /// let vector_field_sum = vector_field_1.add(&vector_field_2).unwrap();
    /// ```
    ///
    /// ## Todo
    /// Improve error handling.
    ///
    pub fn add(
        self: &Self,
        vector_field: &VectorField1D,
    ) -> Result<Self, &'static str> {
        if self.grid != vector_field.grid {
            return Err("Grids of the two vector fields do not match");
        }

        let grid = self.grid.clone();

        // Sums the field values of the two vector fields.
        let field_values: Vec<[f64; 3]> = self
            .field_values
            .iter()
            .zip(vector_field.field_values.iter())
            .map(|(v1, v2)| [v1[0] + v2[0], v1[1] + v2[1], v1[2] + v2[2]])
            .collect();

        Ok(VectorField1D { grid, field_values })
    }

    /// # Subtract vector fields
    ///
    /// ## Description
    /// Subtracts `vector_field` from the current vector field, and returns this
    /// new vector field.
    ///
    /// ## Example use case
    /// ```
    /// let grid = Grid::new_uniform_grid(0.0, 1.0, 11);
    /// let field_values_1 = vec![[1.0, 1.0, 0.0]; grid.grid_points.len()];
    /// let field_values_2 = vec![[0.0, 1.0, 0.0]; grid.grid_points.len()];
    /// let vector_field_1 = VectorField1D::new_vector_field(&grid, field_values_1);
    /// let vector_field_2 = VectorField1D::new_vector_field(&grid, field_values_2);
    /// let vector_field_difference = vector_field_1.subtract(&vector_field_2).unwrap();
    /// ```
    ///
    /// ## Todo
    /// Improve error handling.
    ///
    pub fn subtract(
        self: &Self,
        vector_field: &VectorField1D,
    ) -> Result<Self, &'static str> {
        if self.grid != vector_field.grid {
            return Err("Grids of the two vector fields do not match");
        }

        let grid = self.grid.clone();

        // Subtracts the field values of the two vector fields.
        let field_values: Vec<[f64; 3]> = self
            .field_values
            .iter()
            .zip(vector_field.field_values.iter())
            .map(|(v1, v2)| [v1[0] - v2[0], v1[1] - v2[1], v1[2] - v2[2]])
            .collect();

        Ok(VectorField1D { grid, field_values })
    }

    /// # Scale vector field
    ///
    /// ## Description
    /// Multiplies the current vector field by a scalar, and returns this new
    /// vector field.
    ///
    /// ## Example use case
    /// ```
    /// let grid = Grid::new_uniform_grid(0.0, 1.0, 11);
    /// let field_values = vec![[1.0, 2.0, 3.0]; grid.grid_points.len()];
    /// let vector_field = VectorField1D::new_vector_field(&grid, field_values);
    /// let scaled_vector_field = vector_field.scale(2.0);
    /// ```
    ///
    pub fn scale(self: &Self, scalar: f64) -> Self {
        let grid = self.grid.clone();

        // Scales the field values of the vector field by scalar.
        let field_values: Vec<[f64; 3]> = self
            .field_values
            .iter()
            .map(|v| [v[0] * scalar, v[1] * scalar, v[2] * scalar])
            .collect();

        VectorField1D { grid, field_values }
    }
}

// Vector operations.
impl VectorField1D {
    /// # Dot product
    ///
    /// ## Description
    /// Computes the dot product of the current vector field with
    /// `vector_field`. The result is returned as a scalar field.
    ///
    /// ## Example use case
    /// ```
    /// let grid = Grid::new_uniform_grid(0.0, 1.0, 11);
    /// let field_values_1 = vec![[1.0, 2.0, 3.0]; grid.grid_points.len()];
    /// let field_values_2 = vec![[4.0, 5.0, 6.0]; grid.grid_points.len()];
    /// let vector_field_1 = VectorField1D::new_vector_field(&grid, field_values_1).unwrap();
    /// let vector_field_2 = VectorField1D::new_vector_field(&grid, field_values_2).unwrap();
    /// let new_scalar_field = vector_field_1.dot_product(&vector_field_2);
    /// ```
    ///
    pub fn dot_product(
        self: &Self,
        vector_field: &VectorField1D,
    ) -> Result<ScalarField1D, &'static str> {
        if self.grid != vector_field.grid {
            return Err("Grids of the two vector fields do not match");
        }

        let scalar_field_values: Vec<f64> = self
            .field_values
            .iter()
            .zip(vector_field.field_values.iter())
            .map(|(v1, v2)| v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2])
            .collect();

        Ok(ScalarField1D {
            grid: self.grid.clone(),
            field_values: scalar_field_values,
        })
    }

    /// # Cross product
    ///
    /// ## Description
    /// Computes the cross product of the current vector field with
    /// `vector_field`. The result is returned as a new vector field.
    ///
    /// ## Example use case
    /// ```
    /// let grid = Grid::new_uniform_grid(0.0, 1.0, 11);
    /// let field_values_1 = vec![[1.0, 2.0, 3.0]; grid.grid_points.len()];
    /// let field_values_2 = vec![[4.0, 5.0, 6.0]; grid.grid_points.len()];
    /// let vector_field_1 = VectorField1D::new_vector_field(&grid, field_values_1).unwrap();
    /// let vector_field_2 = VectorField1D::new_vector_field(&grid, field_values_2).unwrap();
    /// let new_vector_field = vector_field_1.cross_product(&vector_field_2);
    /// ```
    ///
    pub fn cross_product(
        self: &Self,
        vector_field: &VectorField1D,
    ) -> Result<Self, &'static str> {
        if self.grid != vector_field.grid {
            return Err("Grids of the two vector fields do not match");
        }

        let grid = self.grid.clone();

        let field_values: Vec<[f64; 3]> = self
            .field_values
            .iter()
            .zip(vector_field.field_values.iter())
            .map(|(v1, v2)| {
                [
                    v1[1] * v2[2] - v1[2] * v2[1],
                    v1[2] * v2[0] - v1[0] * v2[2],
                    v1[0] * v2[1] - v1[1] * v2[0],
                ]
            })
            .collect();

        Ok(VectorField1D { grid, field_values })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::Grid;

    #[test]
    fn test_new_vector_field() {
        let grid = Grid::new_uniform_grid(0.0, 1.0, 11);
        let field_values = vec![[1.0, 0.0, 0.0]; grid.grid_points.len()];
        let vector_field =
            VectorField1D::new_vector_field(&grid, field_values.clone())
                .unwrap();
        assert_eq!(vector_field.grid, grid);
        assert_eq!(vector_field.field_values, field_values);
    }

    #[test]
    fn test_new_constant_vector_field() {
        let grid = Grid::new_uniform_grid(0.0, 1.0, 11);
        let field_value = [1.0, 0.0, 0.0];
        let vector_field =
            VectorField1D::new_constant_vector_field(&grid, field_value);
        assert_eq!(vector_field.grid, grid);
        assert_eq!(
            vector_field.field_values,
            vec![field_value; grid.grid_points.len()]
        );
    }

    #[test]
    fn test_add_vector_fields() {
        let grid = Grid::new_uniform_grid(0.0, 1.0, 11);
        let field_values_1 = vec![[1.0, 0.0, 0.0]; grid.grid_points.len()];
        let field_values_2 = vec![[0.0, 1.0, 0.0]; grid.grid_points.len()];
        let vector_field_1 =
            VectorField1D::new_vector_field(&grid, field_values_1).unwrap();
        let vector_field_2 =
            VectorField1D::new_vector_field(&grid, field_values_2).unwrap();
        let vector_field_sum = vector_field_1.add(&vector_field_2).unwrap();
        let expected_field_values =
            vec![[1.0, 1.0, 0.0]; grid.grid_points.len()];
        assert_eq!(vector_field_sum.field_values, expected_field_values);
    }

    #[test]
    fn test_subtract_vector_fields() {
        let grid = Grid::new_uniform_grid(0.0, 1.0, 11);
        let field_values_1 = vec![[1.0, 1.0, 0.0]; grid.grid_points.len()];
        let field_values_2 = vec![[0.0, 1.0, 0.0]; grid.grid_points.len()];
        let vector_field_1 =
            VectorField1D::new_vector_field(&grid, field_values_1).unwrap();
        let vector_field_2 =
            VectorField1D::new_vector_field(&grid, field_values_2).unwrap();
        let vector_field_difference =
            vector_field_1.subtract(&vector_field_2).unwrap();
        let expected_field_values =
            vec![[1.0, 0.0, 0.0]; grid.grid_points.len()];
        assert_eq!(vector_field_difference.field_values, expected_field_values);
    }

    #[test]
    fn test_scale_vector_field() {
        let grid = Grid::new_uniform_grid(0.0, 1.0, 11);
        let field_values = vec![[1.0, 2.0, 3.0]; grid.grid_points.len()];
        let vector_field =
            VectorField1D::new_vector_field(&grid, field_values.clone())
                .unwrap();
        let scaled_vector_field = vector_field.scale(2.0);
        let expected_field_values =
            vec![[2.0, 4.0, 6.0]; grid.grid_points.len()];
        assert_eq!(scaled_vector_field.field_values, expected_field_values);
    }

    #[test]
    fn test_dot_product() {
        let grid = Grid::new_uniform_grid(0.0, 1.0, 11);
        let field_values_1 = vec![[1.0, 2.0, 3.0]; grid.grid_points.len()];
        let field_values_2 = vec![[4.0, 5.0, 6.0]; grid.grid_points.len()];
        let vector_field_1 =
            VectorField1D::new_vector_field(&grid, field_values_1).unwrap();
        let vector_field_2 =
            VectorField1D::new_vector_field(&grid, field_values_2).unwrap();
        let dot_product_field =
            vector_field_1.dot_product(&vector_field_2).unwrap();
        let expected_values = vec![32.0; grid.grid_points.len()];
        assert_eq!(dot_product_field.field_values, expected_values);
    }

    #[test]
    fn test_cross_product() {
        let grid = Grid::new_uniform_grid(0.0, 1.0, 11);
        let field_values_1 = vec![[1.0, 2.0, 3.0]; grid.grid_points.len()];
        let field_values_2 = vec![[4.0, 5.0, 6.0]; grid.grid_points.len()];
        let vector_field_1 =
            VectorField1D::new_vector_field(&grid, field_values_1).unwrap();
        let vector_field_2 =
            VectorField1D::new_vector_field(&grid, field_values_2).unwrap();
        let cross_product_field =
            vector_field_1.cross_product(&vector_field_2).unwrap();
        let expected_values = vec![[-3.0, 6.0, -3.0]; grid.grid_points.len()];
        assert_eq!(cross_product_field.field_values, expected_values);
    }

    #[test]
    fn test_vector_field_1d_debug() {
        let grid = Grid::new_uniform_grid(0.0, 1.0, 11);
        let field_values = vec![[1.0, 0.0, 0.0]; grid.grid_points.len()];
        let vector_field = VectorField1D::new_vector_field(&grid, field_values);
        let debug_str = format!("{:?}", vector_field);
        assert!(debug_str.contains("VectorField1D"));
        assert!(debug_str.contains("grid"));
        assert!(debug_str.contains("field_values"));
    }

    #[test]
    fn test_vector_field_1d_clone() {
        let grid = Grid::new_uniform_grid(0.0, 1.0, 11);
        let field_values = vec![[1.0, 0.0, 0.0]; grid.grid_points.len()];
        let vector_field =
            VectorField1D::new_vector_field(&grid, field_values.clone());
        let cloned_vector_field = vector_field.clone();
        assert_eq!(vector_field, cloned_vector_field);
    }

    #[test]
    fn test_vector_field_1d_partial_eq() {
        let grid = Grid::new_uniform_grid(0.0, 1.0, 11);
        let field_values_1 = vec![[1.0, 0.0, 0.0]; grid.grid_points.len()];
        let field_values_2 = vec![[0.0, 1.0, 0.0]; grid.grid_points.len()];
        let vector_field_1 =
            VectorField1D::new_vector_field(&grid, field_values_1);
        let vector_field_2 =
            VectorField1D::new_vector_field(&grid, field_values_2);

        // Check vector field is equal to itself.
        assert_eq!(vector_field_1, vector_field_1);

        // Check vector field is equal to a clone.
        assert_eq!(vector_field_1, vector_field_1.clone());

        // Check vector field is not equal to a different vector field.
        assert_ne!(vector_field_1, vector_field_2);
    }

    #[test]
    fn test_add_vector_fields_with_mismatched_grids() {
        let grid1 = Grid::new_uniform_grid(0.0, 1.0, 11);
        let grid2 = Grid::new_uniform_grid(0.0, 2.0, 11);
        let field_values_1 = vec![[1.0, 0.0, 0.0]; grid1.grid_points.len()];
        let field_values_2 = vec![[0.0, 1.0, 0.0]; grid2.grid_points.len()];
        let vector_field_1 =
            VectorField1D::new_vector_field(&grid1, field_values_1).unwrap();
        let vector_field_2 =
            VectorField1D::new_vector_field(&grid2, field_values_2).unwrap();
        let result = vector_field_1.add(&vector_field_2);
        assert!(result.is_err());
        assert_eq!(
            result.err(),
            Some("Grids of the two vector fields do not match")
        );
    }

    #[test]
    fn test_subtract_vector_fields_with_mismatched_grids() {
        let grid1 = Grid::new_uniform_grid(0.0, 1.0, 11);
        let grid2 = Grid::new_uniform_grid(0.0, 2.0, 11);
        let field_values_1 = vec![[1.0, 1.0, 0.0]; grid1.grid_points.len()];
        let field_values_2 = vec![[0.0, 1.0, 0.0]; grid2.grid_points.len()];
        let vector_field_1 =
            VectorField1D::new_vector_field(&grid1, field_values_1).unwrap();
        let vector_field_2 =
            VectorField1D::new_vector_field(&grid2, field_values_2).unwrap();
        let result = vector_field_1.subtract(&vector_field_2);
        assert!(result.is_err());
        assert_eq!(
            result.err(),
            Some("Grids of the two vector fields do not match")
        );
    }

    #[test]
    fn test_dot_product_with_mismatched_grids() {
        let grid1 = Grid::new_uniform_grid(0.0, 1.0, 11);
        let grid2 = Grid::new_uniform_grid(0.0, 2.0, 11);
        let field_values_1 = vec![[1.0, 2.0, 3.0]; grid1.grid_points.len()];
        let field_values_2 = vec![[4.0, 5.0, 6.0]; grid2.grid_points.len()];
        let vector_field_1 =
            VectorField1D::new_vector_field(&grid1, field_values_1).unwrap();
        let vector_field_2 =
            VectorField1D::new_vector_field(&grid2, field_values_2).unwrap();
        let result = vector_field_1.dot_product(&vector_field_2);
        assert!(result.is_err());
        assert_eq!(
            result.err(),
            Some("Grids of the two vector fields do not match")
        );
    }

    #[test]
    fn test_cross_product_with_mismatched_grids() {
        let grid1 = Grid::new_uniform_grid(0.0, 1.0, 11);
        let grid2 = Grid::new_uniform_grid(0.0, 2.0, 11);
        let field_values_1 = vec![[1.0, 2.0, 3.0]; grid1.grid_points.len()];
        let field_values_2 = vec![[4.0, 5.0, 6.0]; grid2.grid_points.len()];
        let vector_field_1 =
            VectorField1D::new_vector_field(&grid1, field_values_1).unwrap();
        let vector_field_2 =
            VectorField1D::new_vector_field(&grid2, field_values_2).unwrap();
        let result = vector_field_1.cross_product(&vector_field_2);
        assert!(result.is_err());
        assert_eq!(
            result.err(),
            Some("Grids of the two vector fields do not match")
        );
    }
}
