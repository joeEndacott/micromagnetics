use crate::{grid::Grid, scalar_field::ScalarField1D};

/// # Vector field 1D
///
/// ## Description
/// A `VectorField1D` represents a 3-vector field sampled on a 1D grid of
/// points. Each grid point has an associated 3-vector, and the Cartesian
/// components of this vector are stored as an array of three `f64` elements.
/// The grid points are represented by a `Grid` instance.
///
/// The x axis is the axis along which the grid points lie. The array associated
/// with each 3-vector is ordered as `[x, y, z]`.
///
/// ## Example use case
/// ```
/// use crate::grid::Grid;
/// use crate::vector_field::VectorField1D;
///
/// let grid = Grid::new_uniform_grid(0.0, 1.0, 11);
/// let field_values = vec![[1.0, 0.0, 0.0]; grid.grid_points.len()];
/// let vector_field = VectorField1D::new_vector_field(&grid, &field_values).unwrap();
/// ```
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
    /// use crate::grid::Grid;
    /// use crate::vector_field::VectorField1D;
    ///
    /// let grid = Grid::new_uniform_grid(0.0, 1.0, 11);
    /// let field_values = vec![[1.0, 0.0, 0.0]; grid.grid_points.len()];
    /// let vector_field = VectorField1D::new_vector_field(&grid, field_values);
    /// ```
    ///
    pub fn new_vector_field(
        grid: &Grid,
        field_values: &Vec<[f64; 3]>,
    ) -> Result<VectorField1D, &'static str> {
        if grid.grid_points.len() != field_values.len() {
            return Err(
                "Number of grid points does not match number of field values",
            );
        }

        Ok(VectorField1D {
            grid: grid.clone(),
            field_values: field_values.clone(),
        })
    }

    /// # New constant vector field
    ///
    /// ## Description
    /// Creates a new 1D 3-vector field with a constant 3D vector at every grid
    /// point.
    ///
    /// ## Example use case
    /// ```
    /// use crate::grid::Grid;
    /// use crate::vector_field::VectorField1D;
    ///
    /// let grid = Grid::new_uniform_grid(0.0, 1.0, 11);
    /// let field_value = [1.0, 0.0, 0.0];
    /// let vector_field = VectorField1D::new_constant_vector_field(&grid, field_value);
    /// ```
    ///
    pub fn new_constant_vector_field(
        grid: &Grid,
        field_value: [f64; 3],
    ) -> VectorField1D {
        let field_values = vec![field_value; grid.grid_points.len()];

        VectorField1D {
            grid: grid.clone(),
            field_values,
        }
    }

    /// # Vector field to scalar fields
    ///
    /// ## Description
    /// Decomposes the current vector field into three scalar fields, each
    /// representing one of the x, y, or z components of the vector field.
    ///
    /// ## Example use case
    /// ```
    /// use crate::grid::Grid;
    /// use crate::vector_field::VectorField1D;
    /// use crate::scalar_field::ScalarField1D;
    ///
    /// let grid = Grid::new_uniform_grid(0.0, 1.0, 11);
    /// let field_values = vec![[1.0, 2.0, 3.0]; grid.grid_points.len()];
    /// let vector_field = VectorField1D::new_vector_field(&grid, field_values).unwrap();
    /// let (vector_field_x_values, vector_field_y_values, vector_field_z_values) = vector_field.scalar_fields_to_vector_field();
    /// ```ß
    ///
    pub fn vector_field_to_scalar_fields(
        self: &Self,
    ) -> (ScalarField1D, ScalarField1D, ScalarField1D) {
        let grid = self.grid.clone();

        let x_values = ScalarField1D {
            grid: grid.clone(),
            field_values: self.field_values.iter().map(|v| v[0]).collect(),
        };

        let y_values = ScalarField1D {
            grid: grid.clone(),
            field_values: self.field_values.iter().map(|v| v[1]).collect(),
        };

        let z_values = ScalarField1D {
            grid: grid.clone(),
            field_values: self.field_values.iter().map(|v| v[2]).collect(),
        };

        (x_values, y_values, z_values)
    }

    /// # Scalar fields to vector field
    ///
    /// ## Description
    /// Combines three scalar fields into a single vector field, where the
    /// scalar fields represent the x, y, and z components of the vector field.
    ///
    /// ## Example use case
    /// ```
    /// use crate::grid::Grid;
    /// use crate::vector_field::VectorField1D;
    /// use crate::scalar_field::ScalarField1D;
    ///
    /// let grid = Grid::new_uniform_grid(0.0, 1.0, 11);
    /// let vector_field_x_values = ScalarField1D::new_constant_scalar_field(&grid, 1.0);
    /// let vector_field_y_values = ScalarField1D::new_constant_scalar_field(&grid, 2.0);
    /// let vector_field_z_values = ScalarField1D::new_constant_scalar_field(&grid, 3.0);
    /// let vector_field = VectorField1D::scalar_fields_to_vector_field((&vector_field_x_values, &vector_field_y_values, &vector_field_z_values));
    /// ```
    ///
    pub fn scalar_fields_to_vector_field(
        (x_values, y_values, z_values): (
            &ScalarField1D,
            &ScalarField1D,
            &ScalarField1D,
        ),
    ) -> Result<Self, &'static str> {
        if x_values.grid != y_values.grid || y_values.grid != z_values.grid {
            return Err("Grids of the scalar fields do not match");
        }

        let field_values: Vec<[f64; 3]> = x_values
            .field_values
            .iter()
            .zip(
                y_values
                    .field_values
                    .iter()
                    .zip(z_values.field_values.iter()),
            )
            .map(|(x, (y, z))| [*x, *y, *z])
            .collect();

        Ok(VectorField1D {
            grid: x_values.grid.clone(),
            field_values,
        })
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
    /// use crate::grid::Grid;
    /// use crate::vector_field::VectorField1D;
    ///
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

        // Sums the field values of the two vector fields.
        let field_values: Vec<[f64; 3]> = self
            .field_values
            .iter()
            .zip(vector_field.field_values.iter())
            .map(|(v1, v2)| [v1[0] + v2[0], v1[1] + v2[1], v1[2] + v2[2]])
            .collect();

        Ok(VectorField1D {
            grid: self.grid.clone(),
            field_values,
        })
    }

    /// # Subtract vector fields
    ///
    /// ## Description
    /// Subtracts `vector_field` from the current vector field, and returns this
    /// new vector field.
    ///
    /// ## Example use case
    /// ```
    /// use crate::grid::Grid;
    /// use crate::vector_field::VectorField1D;
    ///
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

        // Subtracts the field values of the two vector fields.
        let field_values: Vec<[f64; 3]> = self
            .field_values
            .iter()
            .zip(vector_field.field_values.iter())
            .map(|(v1, v2)| [v1[0] - v2[0], v1[1] - v2[1], v1[2] - v2[2]])
            .collect();

        Ok(VectorField1D {
            grid: self.grid.clone(),
            field_values,
        })
    }

    /// # Scale vector field
    ///
    /// ## Description
    /// Multiplies the current vector field by a scalar, and returns this new
    /// vector field.
    ///
    /// ## Example use case
    /// ```
    /// use crate::grid::Grid;
    /// use crate::vector_field::VectorField1D;
    ///
    /// let grid = Grid::new_uniform_grid(0.0, 1.0, 11);
    /// let field_values = vec![[1.0, 2.0, 3.0]; grid.grid_points.len()];
    /// let vector_field = VectorField1D::new_vector_field(&grid, field_values);
    /// let scaled_vector_field = vector_field.scale(2.0);
    /// ```
    ///
    pub fn scale(self: &Self, scalar: f64) -> Self {
        // Scales the field values of the vector field by scalar.
        let field_values: Vec<[f64; 3]> = self
            .field_values
            .iter()
            .map(|v| [v[0] * scalar, v[1] * scalar, v[2] * scalar])
            .collect();

        VectorField1D {
            grid: self.grid.clone(),
            field_values,
        }
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
    /// use crate::grid::Grid;
    /// use crate::vector_field::VectorField1D;
    /// use crate::scalar_field::ScalarField1D;
    ///
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
    /// use crate::grid::Grid;
    /// use crate::vector_field::VectorField1D;
    ///
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

        Ok(VectorField1D {
            grid: self.grid.clone(),
            field_values,
        })
    }
}

// Vector calculus operations
impl VectorField1D {
    /// # Partial derivative with respect to x
    ///
    /// ## Description
    /// `partial_x` computes the partial derivative of the current vector field
    /// with respect to the x coordinate. The result is returned as a new vector
    /// field.
    ///
    /// For a `VectorField1D` instance, the grid points lie along the x axis.
    ///
    /// ## Example use case
    /// ```
    /// use crate::grid::Grid;
    /// use crate::vector_field::VectorField1D;
    ///
    /// let grid = Grid::new_uniform_grid(0.0, 1.0, 11);
    /// let field_values = vec![[1.0, 0.0, 0.0]; grid.grid_points.len()];
    /// let vector_field = VectorField1D::new_vector_field(&grid, field_values).unwrap();
    /// let partial_x_vector_field = vector_field.partial_x();
    /// ```
    ///
    pub fn partial_x(self: &Self) -> Self {
        // Decompose the vector field into three scalar fields.
        let (x_values, y_values, z_values) =
            self.vector_field_to_scalar_fields();

        // Compute the partial derivatives of each scalar field.
        let (partial_x_x_values, partial_x_y_values, partial_x_z_values) = (
            x_values.partial_x(),
            y_values.partial_x(),
            z_values.partial_x(),
        );

        // Combine the partial derivatives into a new vector field.
        let partial_x_vector_field =
            VectorField1D::scalar_fields_to_vector_field((
                &partial_x_x_values,
                &partial_x_y_values,
                &partial_x_z_values,
            ))
            .unwrap();

        VectorField1D {
            grid: self.grid.clone(),
            field_values: partial_x_vector_field.field_values,
        }
    }

    /// # Laplacian
    ///
    /// ## Description
    /// `laplacian` computes the Laplacian of the current vector field. The
    /// result is returned as a new vector field.
    ///
    /// For a 1D vector field defined along the x axis, the Laplacian is
    /// defined as the second partial derivative with respect to x of the
    /// vector field.
    ///
    /// ## Example use case
    /// ```
    /// use crate::grid::Grid;
    /// use crate::vector_field::VectorField1D;
    ///
    /// let grid = Grid::new_uniform_grid(0.0, 1.0, 11);
    /// let field_values = vec![[1.0, 0.0, 0.0]; grid.grid_points.len()];
    /// let vector_field = VectorField1D::new_vector_field(&grid, field_values).unwrap();
    /// let laplacian_vector_field = vector_field.laplacian();
    /// ```
    ///
    pub fn laplacian(self: &Self) -> Self {
        // Compute the second partial derivative of the vector field with
        // respect to x.
        let partial_x_x_vector_field = self.partial_x().partial_x();

        VectorField1D {
            grid: self.grid.clone(),
            field_values: partial_x_x_vector_field.field_values,
        }
    }

    /// # Divergence
    ///
    /// ## Description
    /// `divergence` computes the divergence of the current vector field. The
    /// result is returned as a scalar field.
    ///
    /// For a 1D vector field defined along the x axis, the divergence is
    /// defined as the partial derivative with respect to x of the x component
    /// of the vector field.
    ///
    /// ## Example use case
    /// ```
    /// use crate::grid::Grid;
    /// use crate::vector_field::VectorField1D;
    /// use crate::scalar_field::ScalarField1D;
    ///
    /// let grid = Grid::new_uniform_grid(0.0, 1.0, 11);
    /// let field_values = vec![[1.0, 0.0, 0.0]; grid.grid_points.len()];
    /// let vector_field = VectorField1D::new_vector_field(&grid, field_values).unwrap();
    /// let divergence_scalar_field = vector_field.divergence();
    /// ```
    ///
    pub fn divergence(self: &Self) -> ScalarField1D {
        // Decompose the vector field into three scalar fields.
        let (x_values, _, _) = self.vector_field_to_scalar_fields();

        // Compute the partial derivative of the x component of the vector
        // field.
        let partial_x_values = x_values.partial_x();

        ScalarField1D {
            grid: self.grid.clone(),
            field_values: partial_x_values.field_values,
        }
    }

    /// # Curl
    ///
    /// ## Description
    /// `curl` computes the curl of the current vector field. The result is
    /// returned as a new vector field.
    ///
    /// For a 1D vector field `v = [vx, vy, vz]` defined along the x axis, the
    /// curl of `v` has components `[0, - partial_x(vz), partial_x(vy)]`.
    ///
    /// ## Example use case
    /// ```
    /// use crate::grid::Grid;
    /// use crate::vector_field::VectorField1D;
    ///
    /// let grid = Grid::new_uniform_grid(0.0, 1.0, 11);
    /// let field_values = vec![[1.0, 0.0, 0.0]; grid.grid_points.len()];
    /// let vector_field = VectorField1D::new_vector_field(&grid, field_values).unwrap();
    /// let curl_vector_field = vector_field.curl();
    /// ```
    ///
    pub fn curl(self: &Self) -> Self {
        // Decompose the vector field into three scalar fields.
        let (_, y_values, z_values) = self.vector_field_to_scalar_fields();

        // Compute the partial derivatives of the y and z components of the
        // vector field.
        let partial_x_y_values = y_values.partial_x();
        let partial_x_z_values = z_values.partial_x();

        // Combine the partial derivatives into a new vector field.
        let curl_vector_field = VectorField1D::scalar_fields_to_vector_field((
            &ScalarField1D::new_constant_scalar_field(&self.grid, 0.0),
            &partial_x_z_values.scale(-1.0),
            &partial_x_y_values,
        ))
        .unwrap();

        VectorField1D {
            grid: self.grid.clone(),
            field_values: curl_vector_field.field_values,
        }
    }
}

// Todo: add tests for vector calculus operations.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::Grid;

    #[test]
    fn test_new_vector_field() {
        let grid = Grid::new_uniform_grid(0.0, 1.0, 11);
        let field_values = vec![[1.0, 0.0, 0.0]; grid.grid_points.len()];
        let vector_field =
            VectorField1D::new_vector_field(&grid, &field_values).unwrap();
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
            VectorField1D::new_vector_field(&grid, &field_values_1).unwrap();
        let vector_field_2 =
            VectorField1D::new_vector_field(&grid, &field_values_2).unwrap();
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
            VectorField1D::new_vector_field(&grid, &field_values_1).unwrap();
        let vector_field_2 =
            VectorField1D::new_vector_field(&grid, &field_values_2).unwrap();
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
            VectorField1D::new_vector_field(&grid, &field_values).unwrap();
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
            VectorField1D::new_vector_field(&grid, &field_values_1).unwrap();
        let vector_field_2 =
            VectorField1D::new_vector_field(&grid, &field_values_2).unwrap();
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
            VectorField1D::new_vector_field(&grid, &field_values_1).unwrap();
        let vector_field_2 =
            VectorField1D::new_vector_field(&grid, &field_values_2).unwrap();
        let cross_product_field =
            vector_field_1.cross_product(&vector_field_2).unwrap();
        let expected_values = vec![[-3.0, 6.0, -3.0]; grid.grid_points.len()];
        assert_eq!(cross_product_field.field_values, expected_values);
    }

    #[test]
    fn test_vector_field_1d_debug() {
        let grid = Grid::new_uniform_grid(0.0, 1.0, 11);
        let field_values = vec![[1.0, 0.0, 0.0]; grid.grid_points.len()];
        let vector_field =
            VectorField1D::new_vector_field(&grid, &field_values);
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
            VectorField1D::new_vector_field(&grid, &field_values).unwrap();
        let cloned_vector_field = vector_field.clone();
        assert_eq!(vector_field, cloned_vector_field);
    }

    #[test]
    fn test_vector_field_1d_partial_eq() {
        let grid = Grid::new_uniform_grid(0.0, 1.0, 11);
        let field_values_1 = vec![[1.0, 0.0, 0.0]; grid.grid_points.len()];
        let field_values_2 = vec![[0.0, 1.0, 0.0]; grid.grid_points.len()];
        let vector_field_1 =
            VectorField1D::new_vector_field(&grid, &field_values_1);
        let vector_field_2 =
            VectorField1D::new_vector_field(&grid, &field_values_2);

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
            VectorField1D::new_vector_field(&grid1, &field_values_1).unwrap();
        let vector_field_2 =
            VectorField1D::new_vector_field(&grid2, &field_values_2).unwrap();
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
            VectorField1D::new_vector_field(&grid1, &field_values_1).unwrap();
        let vector_field_2 =
            VectorField1D::new_vector_field(&grid2, &field_values_2).unwrap();
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
            VectorField1D::new_vector_field(&grid1, &field_values_1).unwrap();
        let vector_field_2 =
            VectorField1D::new_vector_field(&grid2, &field_values_2).unwrap();
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
            VectorField1D::new_vector_field(&grid1, &field_values_1).unwrap();
        let vector_field_2 =
            VectorField1D::new_vector_field(&grid2, &field_values_2).unwrap();
        let result = vector_field_1.cross_product(&vector_field_2);
        assert!(result.is_err());
        assert_eq!(
            result.err(),
            Some("Grids of the two vector fields do not match")
        );
    }
}
