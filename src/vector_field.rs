use crate::grid::Grid;

/// # Vector field
///
/// ## Description
/// A `VectorField` represents a 3-vector field sampled on a 1D grid of points.
/// Each grid point has an associated 3-vector, stored as an array of three
/// `f64` elements. The grid points are represented by a `Grid` instance.
///
/// ## Example use case
/// ```
/// use crate::grid::Grid;
/// use crate::vector_field::VectorField;
///
/// // Create a grid.
/// let grid = Grid::new(/* grid parameters */);
///
/// // Create a vector field with specified values.
/// let field_values = vec![[1.0, 0.0, 0.0]; grid.grid_points.len()];
/// let vector_field = VectorField::new_vector_field(&grid, field_values);
///
/// // Create a constant vector field.
/// let constant_vector_field = VectorField::new_constant_vector_field(&grid, [1.0, 0.0, 0.0]);
/// ```
///
#[derive(Debug, Clone, PartialEq)]
pub struct VectorField {
    pub grid: Grid,
    pub field_values: Vec<[f64; 3]>,
}

// Constructor functions.
impl VectorField {
    /// # New vector field
    ///
    /// ## Description
    /// Creates a new vector field, with a specified 3-vector at each grid
    /// point.
    ///
    /// ## Example use case
    /// ```rust
    /// let grid = Grid::new(/* grid parameters */);
    /// let field_values = vec![[1.0, 0.0, 0.0]; grid.grid_points.len()];
    /// let vector_field = VectorField::new_vector_field(&grid, field_values);
    /// ```
    ///
    pub fn new_vector_field(
        grid: &Grid,
        field_values: Vec<[f64; 3]>,
    ) -> VectorField {
        let grid = grid.clone();

        VectorField { grid, field_values }
    }

    /// # New constant vector field
    ///
    /// ## Description
    /// Creates a new vector field with a constant 3D vector at every grid
    /// point.
    ///
    /// ## Example use case
    /// ```rust
    /// let grid = Grid::new(/* grid parameters */);
    /// let constant_vector_field = VectorField::new_constant_vector_field(&grid, [1.0, 0.0, 0.0]);
    /// ```
    ///
    pub fn new_constant_vector_field(
        grid: &Grid,
        field_value: [f64; 3],
    ) -> VectorField {
        let grid = grid.clone();
        let field_values = vec![field_value; grid.grid_points.len()];

        VectorField { grid, field_values }
    }
}

// Arithmetic operations.
impl VectorField {
    /// # Add vector fields
    ///
    /// ## Description
    /// Adds `vector_field` to the current vector field, and returns this new
    /// vector field.
    ///
    /// ## Example use case
    /// ```rust
    /// let vector_field_1 = VectorField::new_vector_field(&grid, field_values_1);
    /// let vector_field_2 = VectorField::new_vector_field(&grid, field_values_2);
    /// let result = vector_field_1.add(&vector_field_2);
    /// ```
    ///
    /// ## Todo
    /// Implement error handling for when the two vector fields have different
    /// `Grids`.
    ///
    pub fn add(self: &Self, vector_field: &VectorField) -> Self {
        let grid = self.grid.clone();

        // Sums the field values of the two vector fields.
        let field_values: Vec<[f64; 3]> = self
            .field_values
            .iter()
            .zip(vector_field.field_values.iter())
            .map(|(v1, v2)| [v1[0] + v2[0], v1[1] + v2[1], v1[2] + v2[2]])
            .collect();

        VectorField { grid, field_values }
    }

    /// # Subtract vector fields
    ///
    /// ## Description
    /// Subtracts `vector_field` from the current vector field, and returns this
    /// new vector field.
    ///
    /// ## Example use case
    /// ```rust
    /// let vector_field_1 = VectorField::new_vector_field(&grid, field_values_1);
    /// let vector_field_2 = VectorField::new_vector_field(&grid, field_values_2);
    /// let result = vector_field_1.subtract(&vector_field_2);
    /// ```
    ///
    /// ## Todo
    /// Implement error handling for when the two vector fields have different
    /// `Grids`.
    ///
    pub fn subtract(self: &Self, vector_field: &VectorField) -> Self {
        let grid = self.grid.clone();

        // Subtracts the field values of the two vector fields.
        let field_values: Vec<[f64; 3]> = self
            .field_values
            .iter()
            .zip(vector_field.field_values.iter())
            .map(|(v1, v2)| [v1[0] - v2[0], v1[1] - v2[1], v1[2] - v2[2]])
            .collect();

        VectorField { grid, field_values }
    }

    /// # Scale vector field
    ///
    /// ## Description
    /// Multiplies the current vector field by a scalar, and returns this new
    /// vector field.
    ///
    /// ## Example use case
    /// ```rust
    /// let vector_field = VectorField::new_vector_field(&grid, field_values);
    /// let scaled_vector_field = vector_field.scale(2.0);
    /// ```
    ///
    pub fn scale(self: &Self, scalar: f64) -> Self {
        let grid = self.grid.clone();

        // Scales the field values of the vector field by the scalar.
        let field_values: Vec<[f64; 3]> = self
            .field_values
            .iter()
            .map(|v| [v[0] * scalar, v[1] * scalar, v[2] * scalar])
            .collect();

        VectorField { grid, field_values }
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
            VectorField::new_vector_field(&grid, field_values.clone());

        assert_eq!(vector_field.grid, grid);
        assert_eq!(vector_field.field_values, field_values);
    }

    #[test]
    fn test_new_constant_vector_field() {
        let grid = Grid::new_uniform_grid(0.0, 1.0, 11);
        let field_value = [1.0, 0.0, 0.0];
        let vector_field =
            VectorField::new_constant_vector_field(&grid, field_value);

        assert_eq!(vector_field.grid, grid);
        assert!(vector_field.field_values.iter().all(|&v| v == field_value));
    }

    #[test]
    fn test_add_vector_fields() {
        let grid = Grid::new_uniform_grid(0.0, 1.0, 11);
        let field_values_1 = vec![[1.0, 0.0, 0.0]; grid.grid_points.len()];
        let field_values_2 = vec![[0.0, 1.0, 0.0]; grid.grid_points.len()];
        let vector_field_1 =
            VectorField::new_vector_field(&grid, field_values_1);
        let vector_field_2 =
            VectorField::new_vector_field(&grid, field_values_2);

        let result = vector_field_1.add(&vector_field_2);
        let expected_values = vec![[1.0, 1.0, 0.0]; grid.grid_points.len()];

        assert_eq!(result.field_values, expected_values);
    }

    #[test]
    fn test_subtract_vector_fields() {
        let grid = Grid::new_uniform_grid(0.0, 1.0, 11);
        let field_values_1 = vec![[1.0, 1.0, 0.0]; grid.grid_points.len()];
        let field_values_2 = vec![[0.0, 1.0, 0.0]; grid.grid_points.len()];
        let vector_field_1 =
            VectorField::new_vector_field(&grid, field_values_1);
        let vector_field_2 =
            VectorField::new_vector_field(&grid, field_values_2);

        let result = vector_field_1.subtract(&vector_field_2);
        let expected_values = vec![[1.0, 0.0, 0.0]; grid.grid_points.len()];

        assert_eq!(result.field_values, expected_values);
    }

    #[test]
    fn test_scale_vector_field() {
        let grid = Grid::new_uniform_grid(0.0, 1.0, 11);
        let field_values = vec![[1.0, 2.0, 3.0]; grid.grid_points.len()];
        let vector_field = VectorField::new_vector_field(&grid, field_values);

        let result = vector_field.scale(2.0);
        let expected_values = vec![[2.0, 4.0, 6.0]; grid.grid_points.len()];

        assert_eq!(result.field_values, expected_values);
    }
}
