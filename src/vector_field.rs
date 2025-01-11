use crate::{
    boundary_conditions::BoundaryConditions1D, grid::Grid,
    scalar_field::ScalarField1D,
};

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
/// The vector field includes details about the boundary conditions that are
/// applied to the field.
///
#[derive(Debug, Clone, PartialEq)]
pub struct VectorField1D {
    pub grid: Grid,
    pub field_values: Vec<[f64; 3]>,
    pub boundary_conditions: BoundaryConditions1D,
}

// Constructor functions.
impl VectorField1D {
    /// # New vector field
    ///
    /// ## Description
    /// Creates a new 1D 3-vector field, with a specified 3-vector at each grid
    /// point.
    ///
    /// The vector field is given the default boundary conditions.
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
            boundary_conditions: BoundaryConditions1D::default(),
        })
    }

    /// # Function to vector field
    ///
    /// ## Description
    /// Generates a `VectorField1D`, given 1D vector valued function and a
    /// `Grid`. The vector valued function is sampled at each grid point and
    /// the values are stored in the field `field_values`.
    ///
    /// The vector field is given the default boundary conditions.
    ///
    pub fn function_to_vector_field<F>(grid: &Grid, func: F) -> Self
    where
        F: Fn(f64) -> [f64; 3],
    {
        // Creates a vector containing the value of func at each grid point.
        let function_values: Vec<[f64; 3]> =
            grid.grid_points.iter().map(|&x| func(x)).collect();

        VectorField1D {
            grid: grid.clone(),
            field_values: function_values,
            boundary_conditions: BoundaryConditions1D::default(),
        }
    }

    /// # New constant vector field
    ///
    /// ## Description
    /// Creates a new 1D 3-vector field with a constant 3D vector at every grid
    /// point.
    ///
    /// The vector field is given the default boundary conditions.
    ///
    pub fn new_constant_vector_field(
        grid: &Grid,
        field_value: [f64; 3],
    ) -> VectorField1D {
        let field_values = vec![field_value; grid.grid_points.len()];

        VectorField1D {
            grid: grid.clone(),
            field_values,
            boundary_conditions: BoundaryConditions1D::default(),
        }
    }

    /// # Vector field to scalar fields
    ///
    /// ## Description
    /// Decomposes the current vector field into three scalar fields, each
    /// representing one of the x, y, or z components of the vector field.
    ///
    /// The scalar fields are given the default boundary conditions.
    ///
    pub fn vector_field_to_scalar_fields(
        self: &Self,
    ) -> Result<[ScalarField1D; 3], &'static str> {
        if self.grid.grid_points.len() != self.field_values.len() {
            return Err(
                "Number of grid points does not match number of field values",
            );
        }

        let vx_values = self.field_values.iter().map(|v| v[0]).collect();
        let vy_values = self.field_values.iter().map(|v| v[1]).collect();
        let vz_values = self.field_values.iter().map(|v| v[2]).collect();

        let mut scalar_field_x =
            ScalarField1D::new_scalar_field(&self.grid, &vx_values)?;
        let mut scalar_field_y =
            ScalarField1D::new_scalar_field(&self.grid, &vy_values)?;
        let mut scalar_field_z =
            ScalarField1D::new_scalar_field(&self.grid, &vz_values)?;

        // Assign correct BCs to each scalar field
        match self.boundary_conditions {
            BoundaryConditions1D::None => {
                // Assign None BCs to each scalar field
                scalar_field_x.boundary_conditions = BoundaryConditions1D::None;
                scalar_field_y.boundary_conditions = BoundaryConditions1D::None;
                scalar_field_z.boundary_conditions = BoundaryConditions1D::None;
            }
            BoundaryConditions1D::Periodic => {
                // Assign PBCs to each scalar field
                scalar_field_x.boundary_conditions =
                    BoundaryConditions1D::Periodic;
                scalar_field_y.boundary_conditions =
                    BoundaryConditions1D::Periodic;
                scalar_field_z.boundary_conditions =
                    BoundaryConditions1D::Periodic;
            }
            BoundaryConditions1D::DirichletVector(
                left_boundary,
                right_boundary,
            ) => {
                scalar_field_x.boundary_conditions =
                    BoundaryConditions1D::DirichletScalar(
                        left_boundary[0],
                        right_boundary[0],
                    );
                scalar_field_y.boundary_conditions =
                    BoundaryConditions1D::DirichletScalar(
                        left_boundary[1],
                        right_boundary[1],
                    );
                scalar_field_z.boundary_conditions =
                    BoundaryConditions1D::DirichletScalar(
                        left_boundary[2],
                        right_boundary[2],
                    );
            }
            BoundaryConditions1D::NeumannVector(
                left_boundary,
                right_boundary,
            ) => {
                scalar_field_x.boundary_conditions =
                    BoundaryConditions1D::NeumannScalar(
                        left_boundary[0],
                        right_boundary[0],
                    );
                scalar_field_y.boundary_conditions =
                    BoundaryConditions1D::NeumannScalar(
                        left_boundary[1],
                        right_boundary[1],
                    );
                scalar_field_z.boundary_conditions =
                    BoundaryConditions1D::NeumannScalar(
                        left_boundary[2],
                        right_boundary[2],
                    );
            }
            _ => {
                return Err(
                    "Boundary conditions not implemented for vector fields",
                );
            }
        }

        Ok([scalar_field_x, scalar_field_y, scalar_field_z])
    }

    /// # Scalar fields to vector field
    ///
    /// ## Description
    /// Combines three scalar fields into a single vector field, where the
    /// scalar fields represent the x, y, and z components of the vector field.
    ///
    /// The scalar fields are represented as an array of three `ScalarField1D`
    /// instances.
    ///
    /// The vector field is given the default boundary conditions.
    ///
    pub fn scalar_fields_to_vector_field(
        scalar_fields: [&ScalarField1D; 3],
    ) -> Result<Self, &'static str> {
        let [x_values, y_values, z_values] = scalar_fields;

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
            boundary_conditions: BoundaryConditions1D::default(),
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
    /// The new vector field is given the default boundary conditions.
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
            boundary_conditions: BoundaryConditions1D::default(),
        })
    }

    /// # Subtract vector fields
    ///
    /// ## Description
    /// Subtracts `vector_field` from the current vector field, and returns this
    /// new vector field.
    ///
    /// The new vector field is given the default boundary conditions.
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
            boundary_conditions: BoundaryConditions1D::default(),
        })
    }

    /// # Scale vector field
    ///
    /// ## Description
    /// Multiplies the current vector field by a scalar, and returns this new
    /// vector field.
    ///
    /// The new vector field is given the default boundary conditions.
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
            boundary_conditions: BoundaryConditions1D::default(),
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
    /// The new scalar field is given the default boundary conditions.
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
            boundary_conditions: BoundaryConditions1D::default(),
        })
    }

    /// # Cross product
    ///
    /// ## Description
    /// Computes the cross product of the current vector field with
    /// `vector_field`. The result is returned as a new vector field.
    ///
    /// The new vector field is given the default boundary conditions.
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
            boundary_conditions: BoundaryConditions1D::default(),
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
    /// The new vector field is given the default boundary conditions.
    ///
    /// For a `VectorField1D` instance, the grid points lie along the x axis.
    ///
    pub fn partial_x(self: &Self) -> Result<Self, &'static str> {
        // Decompose the vector field into three scalar fields.
        let [vx, vy, vz] = self.vector_field_to_scalar_fields()?;

        // Compute the partial derivatives of each scalar field.
        let [partial_x_vx, partial_x_vy, partial_x_vz] =
            [vx.partial_x()?, vy.partial_x()?, vz.partial_x()?];

        // Combine the partial derivatives into a new vector field.
        let partial_x_vector_field =
            VectorField1D::scalar_fields_to_vector_field([
                &partial_x_vx,
                &partial_x_vy,
                &partial_x_vz,
            ])?;

        Ok(VectorField1D {
            grid: self.grid.clone(),
            field_values: partial_x_vector_field.field_values,
            boundary_conditions: BoundaryConditions1D::default(),
        })
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
    pub fn laplacian(self: &Self) -> Result<Self, &'static str> {
        // Compute the second partial derivative of the vector field with
        // respect to x.
        self.partial_x()?.partial_x()
    }

    /// # Divergence
    ///
    /// ## Description
    /// `divergence` computes the divergence of the current vector field. The
    /// result is returned as a scalar field.
    ///
    /// The scalar field is given the default boundary conditions.
    ///
    /// For a 1D vector field defined along the x axis, the divergence is
    /// defined as the partial derivative with respect to x of the x component
    /// of the vector field.
    ///
    /// ## Todo
    /// Implement a new function which directly calculates the second partial
    /// derivative of the vector field with respect to x, rather than
    /// calculating a first derivative twice.
    ///
    pub fn divergence(self: &Self) -> Result<ScalarField1D, &'static str> {
        // Decompose the vector field into three scalar fields.
        let [vx, _, _] = self.vector_field_to_scalar_fields()?;

        // Compute the partial derivative of the x component of the vector
        // field.
        let partial_x_vx = vx.partial_x()?;

        Ok(ScalarField1D {
            grid: self.grid.clone(),
            field_values: partial_x_vx.field_values,
            boundary_conditions: BoundaryConditions1D::default(),
        })
    }

    /// # Curl
    ///
    /// ## Description
    /// `curl` computes the curl of the current vector field. The result is
    /// returned as a new vector field.
    ///
    /// The new vector field is given the default boundary conditions.
    ///
    /// For a 1D vector field `v = [vx, vy, vz]` defined along the x axis, the
    /// curl of `v` has components `[0, - partial_x(vz), partial_x(vy)]`.
    ///
    pub fn curl(self: &Self) -> Result<Self, &'static str> {
        // Decompose the vector field into three scalar fields.
        let [_, y_values, z_values] = self.vector_field_to_scalar_fields()?;

        // Compute the partial derivatives of the y and z components of the
        // vector field.
        let partial_x_y_values = y_values.partial_x()?;
        let partial_x_z_values = z_values.partial_x()?;

        // Combine the partial derivatives into a new vector field and return
        // this field.
        Ok(VectorField1D::scalar_fields_to_vector_field([
            &ScalarField1D::new_constant_scalar_field(&self.grid, 0.0),
            &partial_x_z_values.scale(-1.0),
            &partial_x_y_values,
        ])?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{grid::Grid, utils};

    mod traits {
        use super::*;

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
    }

    mod constructor_functions {
        use super::*;

        #[test]
        fn test_new_vector_field() {
            let grid = Grid::new_uniform_grid(0.0, 1.0, 11);
            let field_values = vec![[1.0, 0.0, 0.0]; grid.grid_points.len()];
            let vector_field =
                VectorField1D::new_vector_field(&grid, &field_values).unwrap();
            assert_eq!(vector_field.grid, grid);
            assert_eq!(vector_field.field_values, field_values);
            assert_eq!(
                vector_field.boundary_conditions,
                BoundaryConditions1D::default()
            );
        }

        #[test]
        fn test_new_vector_field_error() {
            let grid = Grid::new_uniform_grid(0.0, 1.0, 11);
            let field_values =
                vec![[1.0, 0.0, 0.0]; grid.grid_points.len() - 1];
            let result = VectorField1D::new_vector_field(&grid, &field_values);
            assert!(result.is_err());
            assert_eq!(
            result.err(),
            Some("Number of grid points does not match number of field values")
        );
        }

        #[test]
        fn test_function_to_vector_field() {
            let grid = Grid::new_uniform_grid(0.0, 1.0, 11);
            let vector_field =
                VectorField1D::function_to_vector_field(&grid, |x| {
                    [x, x.powi(2), x.powi(3)]
                });

            let expected_field_values: Vec<[f64; 3]> = grid
                .grid_points
                .iter()
                .map(|&x| [x, x.powi(2), x.powi(3)])
                .collect();
            let expected_vector_field = VectorField1D {
                grid: grid.clone(),
                field_values: expected_field_values,
                boundary_conditions: BoundaryConditions1D::default(),
            };

            assert_eq!(vector_field, expected_vector_field);
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
            assert_eq!(
                vector_field.boundary_conditions,
                BoundaryConditions1D::default()
            );
        }

        #[test]
        fn test_vector_field_to_scalar_fields() {
            let grid = Grid::new_uniform_grid(0.0, 1.0, 11);
            let vector_field = VectorField1D::new_constant_vector_field(
                &grid,
                [1.0, 2.0, 3.0],
            );
            let [x_values, y_values, z_values] =
                vector_field.vector_field_to_scalar_fields().unwrap();

            let expected_x_scalar_field =
                ScalarField1D::new_constant_scalar_field(&grid, 1.0);
            let expected_y_scalar_field =
                ScalarField1D::new_constant_scalar_field(&grid, 2.0);
            let expected_z_scalar_field =
                ScalarField1D::new_constant_scalar_field(&grid, 3.0);

            assert_eq!(x_values, expected_x_scalar_field);
            assert_eq!(y_values, expected_y_scalar_field);
            assert_eq!(z_values, expected_z_scalar_field);
        }

        #[test]
        fn test_scalar_fields_to_vector_field() {
            let grid = Grid::new_uniform_grid(0.0, 1.0, 11);
            let x_values = ScalarField1D::new_constant_scalar_field(&grid, 1.0);
            let y_values = ScalarField1D::new_constant_scalar_field(&grid, 2.0);
            let z_values = ScalarField1D::new_constant_scalar_field(&grid, 3.0);
            let vector_field = VectorField1D::scalar_fields_to_vector_field([
                &x_values, &y_values, &z_values,
            ])
            .unwrap();

            let expected_vector_field =
                VectorField1D::new_constant_vector_field(
                    &grid,
                    [1.0, 2.0, 3.0],
                );
            assert_eq!(vector_field, expected_vector_field);
        }
    }

    mod arithmetic_operations {
        use super::*;

        #[test]
        fn test_add_vector_fields() {
            let grid = Grid::new_uniform_grid(0.0, 1.0, 11);
            let field_values_1 = vec![[1.0, 0.0, 0.0]; grid.grid_points.len()];
            let field_values_2 = vec![[0.0, 1.0, 0.0]; grid.grid_points.len()];
            let vector_field_1 =
                VectorField1D::new_vector_field(&grid, &field_values_1)
                    .unwrap();
            let vector_field_2 =
                VectorField1D::new_vector_field(&grid, &field_values_2)
                    .unwrap();
            let vector_field_sum = vector_field_1.add(&vector_field_2).unwrap();
            let expected_field_values =
                vec![[1.0, 1.0, 0.0]; grid.grid_points.len()];
            assert_eq!(vector_field_sum.field_values, expected_field_values);
        }

        #[test]
        fn test_add_vector_fields_with_mismatched_grids() {
            let grid1 = Grid::new_uniform_grid(0.0, 1.0, 11);
            let grid2 = Grid::new_uniform_grid(0.0, 2.0, 11);
            let field_values_1 = vec![[1.0, 0.0, 0.0]; grid1.grid_points.len()];
            let field_values_2 = vec![[0.0, 1.0, 0.0]; grid2.grid_points.len()];
            let vector_field_1 =
                VectorField1D::new_vector_field(&grid1, &field_values_1)
                    .unwrap();
            let vector_field_2 =
                VectorField1D::new_vector_field(&grid2, &field_values_2)
                    .unwrap();
            let result = vector_field_1.add(&vector_field_2);
            assert!(result.is_err());
            assert_eq!(
                result.err(),
                Some("Grids of the two vector fields do not match")
            );
        }

        #[test]
        fn test_subtract_vector_fields() {
            let grid = Grid::new_uniform_grid(0.0, 1.0, 11);
            let field_values_1 = vec![[1.0, 1.0, 0.0]; grid.grid_points.len()];
            let field_values_2 = vec![[0.0, 1.0, 0.0]; grid.grid_points.len()];
            let vector_field_1 =
                VectorField1D::new_vector_field(&grid, &field_values_1)
                    .unwrap();
            let vector_field_2 =
                VectorField1D::new_vector_field(&grid, &field_values_2)
                    .unwrap();
            let vector_field_difference =
                vector_field_1.subtract(&vector_field_2).unwrap();
            let expected_field_values =
                vec![[1.0, 0.0, 0.0]; grid.grid_points.len()];
            assert_eq!(
                vector_field_difference.field_values,
                expected_field_values
            );
        }

        #[test]
        fn test_subtract_vector_fields_with_mismatched_grids() {
            let grid1 = Grid::new_uniform_grid(0.0, 1.0, 11);
            let grid2 = Grid::new_uniform_grid(0.0, 2.0, 11);
            let field_values_1 = vec![[1.0, 1.0, 0.0]; grid1.grid_points.len()];
            let field_values_2 = vec![[0.0, 1.0, 0.0]; grid2.grid_points.len()];
            let vector_field_1 =
                VectorField1D::new_vector_field(&grid1, &field_values_1)
                    .unwrap();
            let vector_field_2 =
                VectorField1D::new_vector_field(&grid2, &field_values_2)
                    .unwrap();
            let result = vector_field_1.subtract(&vector_field_2);
            assert!(result.is_err());
            assert_eq!(
                result.err(),
                Some("Grids of the two vector fields do not match")
            );
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
    }

    mod vector_operations {
        use super::*;

        #[test]
        fn test_dot_product() {
            let grid = Grid::new_uniform_grid(0.0, 1.0, 11);
            let field_values_1 = vec![[1.0, 2.0, 3.0]; grid.grid_points.len()];
            let field_values_2 = vec![[4.0, 5.0, 6.0]; grid.grid_points.len()];
            let vector_field_1 =
                VectorField1D::new_vector_field(&grid, &field_values_1)
                    .unwrap();
            let vector_field_2 =
                VectorField1D::new_vector_field(&grid, &field_values_2)
                    .unwrap();
            let dot_product_field =
                vector_field_1.dot_product(&vector_field_2).unwrap();
            let expected_values = vec![32.0; grid.grid_points.len()];
            assert_eq!(dot_product_field.field_values, expected_values);
        }

        #[test]
        fn test_dot_product_with_mismatched_grids() {
            let grid1 = Grid::new_uniform_grid(0.0, 1.0, 11);
            let grid2 = Grid::new_uniform_grid(0.0, 2.0, 11);
            let field_values_1 = vec![[1.0, 2.0, 3.0]; grid1.grid_points.len()];
            let field_values_2 = vec![[4.0, 5.0, 6.0]; grid2.grid_points.len()];
            let vector_field_1 =
                VectorField1D::new_vector_field(&grid1, &field_values_1)
                    .unwrap();
            let vector_field_2 =
                VectorField1D::new_vector_field(&grid2, &field_values_2)
                    .unwrap();
            let result = vector_field_1.dot_product(&vector_field_2);
            assert!(result.is_err());
            assert_eq!(
                result.err(),
                Some("Grids of the two vector fields do not match")
            );
        }

        #[test]
        fn test_cross_product() {
            let grid = Grid::new_uniform_grid(0.0, 1.0, 11);
            let field_values_1 = vec![[1.0, 2.0, 3.0]; grid.grid_points.len()];
            let field_values_2 = vec![[4.0, 5.0, 6.0]; grid.grid_points.len()];
            let vector_field_1 =
                VectorField1D::new_vector_field(&grid, &field_values_1)
                    .unwrap();
            let vector_field_2 =
                VectorField1D::new_vector_field(&grid, &field_values_2)
                    .unwrap();
            let cross_product_field =
                vector_field_1.cross_product(&vector_field_2).unwrap();
            let expected_values =
                vec![[-3.0, 6.0, -3.0]; grid.grid_points.len()];
            assert_eq!(cross_product_field.field_values, expected_values);
        }

        #[test]
        fn test_cross_product_with_mismatched_grids() {
            let grid1 = Grid::new_uniform_grid(0.0, 1.0, 11);
            let grid2 = Grid::new_uniform_grid(0.0, 2.0, 11);
            let field_values_1 = vec![[1.0, 2.0, 3.0]; grid1.grid_points.len()];
            let field_values_2 = vec![[4.0, 5.0, 6.0]; grid2.grid_points.len()];
            let vector_field_1 =
                VectorField1D::new_vector_field(&grid1, &field_values_1)
                    .unwrap();
            let vector_field_2 =
                VectorField1D::new_vector_field(&grid2, &field_values_2)
                    .unwrap();
            let result = vector_field_1.cross_product(&vector_field_2);
            assert!(result.is_err());
            assert_eq!(
                result.err(),
                Some("Grids of the two vector fields do not match")
            );
        }
    }

    mod vector_calculus_operations {
        use super::*;

        #[test]
        fn test_partial_x() {
            let grid = Grid::new_uniform_grid(0.0, 1.0, 100);

            // Test the derivative of a constant vector field.
            let v_x = ScalarField1D::new_constant_scalar_field(&grid, 1.0);
            let v_y = ScalarField1D::new_constant_scalar_field(&grid, 3.0);
            let v_z = ScalarField1D::new_constant_scalar_field(&grid, -6.0);
            let vector_field = VectorField1D::scalar_fields_to_vector_field([
                &v_x, &v_y, &v_z,
            ])
            .unwrap();
            let partial_x_vector_field = vector_field.partial_x().unwrap();

            let expected_result = VectorField1D::new_constant_vector_field(
                &grid,
                [0.0, 0.0, 0.0],
            );

            assert!(utils::vector_field_equality(
                &expected_result,
                &partial_x_vector_field,
                1e-6
            ));

            // Test the derivative of the vector field [x, -2x, 4x].
            let vx = ScalarField1D::function_to_scalar_field(&grid, |x| x);
            let vy =
                ScalarField1D::function_to_scalar_field(&grid, |x| -2.0 * x);
            let vz =
                ScalarField1D::function_to_scalar_field(&grid, |x| 4.0 * x);
            let vector_field =
                VectorField1D::scalar_fields_to_vector_field([&vx, &vy, &vz])
                    .unwrap();
            let partial_x_vector_field = vector_field.partial_x().unwrap();

            let expected_result = VectorField1D::new_constant_vector_field(
                &grid,
                [1.0, -2.0, 4.0],
            );

            assert!(utils::vector_field_equality(
                &expected_result,
                &partial_x_vector_field,
                1e-6
            ));
        }

        #[test]
        fn test_laplacian() {
            let grid = Grid::new_uniform_grid(0.0, 1.0, 100);

            // Test the Laplacian of a constant vector field.
            let v_x = ScalarField1D::new_constant_scalar_field(&grid, 1.0);
            let v_y = ScalarField1D::new_constant_scalar_field(&grid, 3.0);
            let v_z = ScalarField1D::new_constant_scalar_field(&grid, -6.0);
            let vector_field = VectorField1D::scalar_fields_to_vector_field([
                &v_x, &v_y, &v_z,
            ])
            .unwrap();
            let laplacian_vector_field = vector_field.laplacian().unwrap();

            let expected_result = VectorField1D::new_constant_vector_field(
                &grid,
                [0.0, 0.0, 0.0],
            );

            assert!(utils::vector_field_equality(
                &expected_result,
                &laplacian_vector_field,
                1e-6
            ));

            // Test the Laplacian of the vector field [x^2, -2x^2, 4x^2].
            let vx = ScalarField1D::function_to_scalar_field(&grid, |x| x * x);
            let vy = ScalarField1D::function_to_scalar_field(&grid, |x| {
                -2.0 * x * x
            });
            let vz =
                ScalarField1D::function_to_scalar_field(&grid, |x| 4.0 * x * x);
            let vector_field =
                VectorField1D::scalar_fields_to_vector_field([&vx, &vy, &vz])
                    .unwrap();
            let laplacian_vector_field = vector_field.laplacian().unwrap();

            let expected_result = VectorField1D::new_constant_vector_field(
                &grid,
                [2.0, -4.0, 8.0],
            );

            assert!(utils::vector_field_equality(
                &expected_result,
                &laplacian_vector_field,
                1e-6
            ));
        }

        #[test]
        fn test_divergence() {
            let grid = Grid::new_uniform_grid(0.0, 1.0, 100);

            // Test the divergence of a constant vector field.
            let vector_field = VectorField1D::new_constant_vector_field(
                &grid,
                [1.0, 3.0, -6.0],
            );
            let divergence_scalar_field = vector_field.divergence().unwrap();

            let expected_result =
                ScalarField1D::new_constant_scalar_field(&grid, 0.0);

            assert!(utils::scalar_field_equality(
                &expected_result,
                &divergence_scalar_field,
                1e-6
            ));

            // Test the divergence of the vector field [x, 0, 0].
            let vx = ScalarField1D::function_to_scalar_field(&grid, |x| x);
            let vy = ScalarField1D::new_constant_scalar_field(&grid, 0.0);
            let vz = ScalarField1D::new_constant_scalar_field(&grid, 0.0);
            let vector_field =
                VectorField1D::scalar_fields_to_vector_field([&vx, &vy, &vz])
                    .unwrap();
            let divergence_scalar_field = vector_field.divergence().unwrap();

            let expected_result =
                ScalarField1D::new_constant_scalar_field(&grid, 1.0);

            assert!(utils::scalar_field_equality(
                &expected_result,
                &divergence_scalar_field,
                1e-6
            ));

            // // Test the divergence of the vector field [2*x, x^3, x^7].
            let vx =
                ScalarField1D::function_to_scalar_field(&grid, |x| 2.0 * x);
            let vy =
                ScalarField1D::function_to_scalar_field(&grid, |x| x.powi(3));
            let vz =
                ScalarField1D::function_to_scalar_field(&grid, |x| x.powi(7));
            let vector_field =
                VectorField1D::scalar_fields_to_vector_field([&vx, &vy, &vz])
                    .unwrap();
            let divergence_scalar_field = vector_field.divergence().unwrap();

            let expected_result =
                ScalarField1D::new_constant_scalar_field(&grid, 2.0);

            assert!(utils::scalar_field_equality(
                &expected_result,
                &divergence_scalar_field,
                1e-6
            ));

            // Test the divergence of the vector field [x^2, x, x].
            let vx = ScalarField1D::function_to_scalar_field(&grid, |x| x * x);
            let vy = ScalarField1D::function_to_scalar_field(&grid, |x| x);
            let vz = ScalarField1D::function_to_scalar_field(&grid, |x| x);
            let vector_field =
                VectorField1D::scalar_fields_to_vector_field([&vx, &vy, &vz])
                    .unwrap();
            let divergence_scalar_field = vector_field.divergence().unwrap();

            let expected_result =
                ScalarField1D::function_to_scalar_field(&grid, |x| 2.0 * x);

            assert!(utils::scalar_field_equality(
                &expected_result,
                &divergence_scalar_field,
                1e-6
            ));
        }

        #[test]
        fn test_curl() {
            let grid = Grid::new_uniform_grid(0.0, 1.0, 100);

            // Test the curl of a constant vector field.
            let vector_field = VectorField1D::new_constant_vector_field(
                &grid,
                [1.0, 3.0, -6.0],
            );
            let curl_vector_field = vector_field.curl().unwrap();

            let expected_result = VectorField1D::new_constant_vector_field(
                &grid,
                [0.0, 0.0, 0.0],
            );

            assert!(utils::vector_field_equality(
                &expected_result,
                &curl_vector_field,
                1e-6
            ));

            // Test the curl of the vector field [x^2, x, -x].
            let vx =
                ScalarField1D::function_to_scalar_field(&grid, |x| x.powi(2));
            let vy = ScalarField1D::function_to_scalar_field(&grid, |x| x);
            let vz = ScalarField1D::function_to_scalar_field(&grid, |x| -x);
            let vector_field =
                VectorField1D::scalar_fields_to_vector_field([&vx, &vy, &vz])
                    .unwrap();
            let curl_vector_field = vector_field.curl().unwrap();

            let expected_result = VectorField1D::new_constant_vector_field(
                &grid,
                [0.0, 1.0, 1.0],
            );

            assert!(utils::vector_field_equality(
                &expected_result,
                &curl_vector_field,
                1e-6
            ));

            // Test the curl of the vector field [x, x^2, -x^2].
            let vx = ScalarField1D::function_to_scalar_field(&grid, |x| x);
            let vy = ScalarField1D::function_to_scalar_field(&grid, |x| x * x);
            let vz = ScalarField1D::function_to_scalar_field(&grid, |x| -x * x);
            let vector_field =
                VectorField1D::scalar_fields_to_vector_field([&vx, &vy, &vz])
                    .unwrap();
            let curl_vector_field = vector_field.curl().unwrap();

            let expected_result =
                VectorField1D::function_to_vector_field(&grid, |x| {
                    [0.0, 2.0 * x, 2.0 * x]
                });

            assert!(utils::vector_field_equality(
                &expected_result,
                &curl_vector_field,
                1e-6
            ));
        }
    }
}
