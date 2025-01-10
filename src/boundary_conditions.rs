use crate::scalar_field::ScalarField1D;
use crate::utils;
use crate::vector_field::VectorField1D;

/// # Boundary conditions
///
/// ## Description
/// The `BoundaryConditions` enum represents the boundary conditions (BCs) that
/// can be applied to a vector field.
///
/// The BCs can be one of the following:
/// - `None`: No boundary conditions are applied.
/// - `Periodic`: Periodic boundary conditions are applied.
/// - `Dirichlet`: Dirichlet boundary conditions are applied. The boundary
/// values are specified as an array of three `f64` elements.
/// - `Neumann`: Neumann boundary conditions are applied. The boundary values
/// are specified as an array of three `f64` elements.
///
#[derive(Debug, Clone, PartialEq)]
pub enum BoundaryConditions1D {
    None,
    Periodic,
    DirichletScalar(f64, f64),
    DirichletVector([f64; 3], [f64; 3]),
    NeumannScalar(f64, f64),
    NeumannVector([f64; 3], [f64; 3]),
}

// Default value of BoundaryConditions.
impl Default for BoundaryConditions1D {
    fn default() -> Self {
        BoundaryConditions1D::None
    }
}

// Constructor functions.
impl BoundaryConditions1D {
    /// # New none boundary conditions
    ///
    /// ## Description
    /// Creates a new `BoundaryConditions` instance with no boundary conditions.
    ///
    pub fn new_no_bcs() -> Self {
        BoundaryConditions1D::None
    }

    /// # New periodic boundary conditions
    ///
    /// ## Description
    /// Creates a new `BoundaryConditions` instance with periodic boundary
    /// conditions.
    ///
    pub fn new_periodic_bcs() -> Self {
        BoundaryConditions1D::Periodic
    }

    /// # New Dirichlet scalar boundary conditions
    ///
    /// ## Description
    /// Creates a new `BoundaryConditions` instance with Dirichlet scalar
    /// boundary conditions. The value of the vector field at the left and
    /// right edges of the domain are specified as `f64` elements.
    ///
    pub fn new_dirichlet_scalar_bcs(
        left_boundary_value: f64,
        right_boundary_value: f64,
    ) -> Self {
        BoundaryConditions1D::DirichletScalar(
            left_boundary_value,
            right_boundary_value,
        )
    }

    /// # New Dirichlet vector boundary conditions
    ///
    /// ## Description
    /// Creates a new `BoundaryConditions` instance with Dirichlet vector
    /// boundary conditions. The value of the vector field at the left and
    /// right edges of the domain are specified as arrays of three `f64`
    /// elements.
    ///
    pub fn new_dirichlet_vector_bcs(
        left_boundary_value: [f64; 3],
        right_boundary_value: [f64; 3],
    ) -> Self {
        BoundaryConditions1D::DirichletVector(
            left_boundary_value,
            right_boundary_value,
        )
    }

    /// # New Neumann scalar boundary conditions
    ///
    /// ## Description
    /// Creates a new `BoundaryConditions` instance with Neumann scalar boundary
    /// conditions. The value of the normal derivative of the vector field at
    /// the left and right edges of the domain are specified as `f64` elements.
    ///
    pub fn new_neumann_scalar_bcs(
        left_boundary_value: f64,
        right_boundary_value: f64,
    ) -> Self {
        BoundaryConditions1D::NeumannScalar(
            left_boundary_value,
            right_boundary_value,
        )
    }

    /// # New Neumann vector boundary conditions
    ///
    /// ## Description
    /// Creates a new `BoundaryConditions` instance with Neumann vector boundary
    /// conditions. The value of the normal derivative of the vector field at
    /// the left and right edges of the domain are specified as arrays of three
    /// `f64` elements.
    ///
    pub fn new_neumann_vector_bcs(
        left_boundary_value: [f64; 3],
        right_boundary_value: [f64; 3],
    ) -> Self {
        BoundaryConditions1D::NeumannVector(
            left_boundary_value,
            right_boundary_value,
        )
    }
}

// Check and boundary conditions.
impl BoundaryConditions1D {
    /// # Check boundary conditions are satisfied for a scalar field
    ///
    /// ## Description
    /// Checks that the boundary conditions are satisfied for a scalar field
    /// within a set tolerance.
    ///
    /// If the boundary conditions are satisfied, returns `Ok(())`, else returns
    /// an error message.
    ///
    pub fn check_scalar_bcs(
        scalar_field: &ScalarField1D,
        tolerance: f64,
    ) -> Result<(), &'static str> {
        let num_points = scalar_field.field_values.len();
        if num_points < 2 {
            return Err("At least two grid points are required");
        }

        match scalar_field.boundary_conditions {
            BoundaryConditions1D::None => Ok(()),
            BoundaryConditions1D::Periodic => {
                if !utils::scalar_equality(
                    scalar_field.field_values[0],
                    scalar_field.field_values[num_points - 1],
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
                    scalar_field.field_values[0],
                    left_boundary_value,
                    tolerance,
                ) || !utils::scalar_equality(
                    scalar_field.field_values[num_points - 1],
                    right_boundary_value,
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

    /// # Check boundary conditions are satisfied for a vector field
    ///
    /// ## Description
    /// Checks that the boundary conditions are satisfied for a vector field
    /// within a set tolerance.
    ///
    /// If the boundary conditions are satisfied, returns `Ok(())`, else returns
    /// an error message.
    ///
    pub fn check_vector_bcs(
        vector_field: &VectorField1D,
        tolerance: f64,
    ) -> Result<(), &'static str> {
        let num_points = vector_field.field_values.len();
        if num_points < 2 {
            return Err("At least two grid points are required");
        }

        match vector_field.boundary_conditions {
            BoundaryConditions1D::None => Ok(()),
            BoundaryConditions1D::Periodic => {
                if !utils::vector_equality(
                    vector_field.field_values[0],
                    vector_field.field_values[num_points - 1],
                    tolerance,
                ) {
                    return Err("Periodic BCs: Field values do not match at the boundaries");
                }
                Ok(())
            }
            BoundaryConditions1D::DirichletVector(
                left_boundary_value,
                right_boundary_value,
            ) => {
                if !utils::vector_equality(
                    vector_field.field_values[0],
                    left_boundary_value,
                    tolerance,
                ) || !utils::vector_equality(
                    vector_field.field_values[num_points - 1],
                    right_boundary_value,
                    tolerance,
                ) {
                    return Err("Dirichlet BCs: Field values do not match specified boundary values");
                }
                Ok(())
            }
            BoundaryConditions1D::NeumannVector(_, _) => {
                // Implement Neumann BC logic.
                Ok(())
            }
            BoundaryConditions1D::DirichletScalar(_, _)
            | BoundaryConditions1D::NeumannScalar(_, _) => {
                return Err("Scalar BCs are not supported for vector fields");
            }
        }
    }

    pub fn apply_scalar_bcs(scalar_field: &mut ScalarField1D) {
        let num_points = scalar_field.field_values.len();
        if num_points < 2 {
            return;
        }

        match scalar_field.boundary_conditions {
            BoundaryConditions1D::None => {}
            BoundaryConditions1D::Periodic => {
                scalar_field.field_values[0] =
                    scalar_field.field_values[num_points - 1];
            }
            BoundaryConditions1D::DirichletScalar(
                left_boundary_value,
                right_boundary_value,
            ) => {
                scalar_field.field_values[0] = left_boundary_value;
                scalar_field.field_values[num_points - 1] =
                    right_boundary_value;
            }
            BoundaryConditions1D::NeumannScalar(_, _) => {
                // Implement Neumann BC logic.
            }
            BoundaryConditions1D::DirichletVector(_, _)
            | BoundaryConditions1D::NeumannVector(_, _) => {
                panic!("Vector BCs are not supported for scalar fields");
            }
        }
    }

    pub fn apply_vector_bcs(vector_field: &mut VectorField1D) {
        let num_points = vector_field.field_values.len();
        if num_points < 2 {
            return;
        }

        match vector_field.boundary_conditions {
            BoundaryConditions1D::None => {}
            BoundaryConditions1D::Periodic => {
                vector_field.field_values[0] =
                    vector_field.field_values[num_points - 1];
            }
            BoundaryConditions1D::DirichletVector(
                left_boundary_value,
                right_boundary_value,
            ) => {
                vector_field.field_values[0] = left_boundary_value;
                vector_field.field_values[num_points - 1] =
                    right_boundary_value;
            }
            BoundaryConditions1D::NeumannVector(_, _) => {
                // Implement Neumann BC logic.
            }
            BoundaryConditions1D::DirichletScalar(_, _)
            | BoundaryConditions1D::NeumannScalar(_, _) => {
                panic!("Scalar BCs are not supported for vector fields");
            }
        }
    }
}

// Todo: add tests for all functions.
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_boundary_conditions() {
        let default_boundary_conditions: BoundaryConditions1D =
            Default::default();
        assert_eq!(default_boundary_conditions, BoundaryConditions1D::None);
    }

    #[test]
    fn test_boundary_conditions_debug() {
        // Debug string for None boundary conditions.
        let boundary_conditions = BoundaryConditions1D::None;
        let debug_str = format!("{:?}", boundary_conditions);
        assert!(debug_str.contains("None"));

        // Debug string for periodic boundary conditions.
        let boundary_conditions = BoundaryConditions1D::Periodic;
        let debug_str = format!("{:?}", boundary_conditions);
        assert!(debug_str.contains("Periodic"));

        // Debug string for Dirichlet scalar boundary conditions.
        let boundary_conditions =
            BoundaryConditions1D::DirichletScalar(0.0, 0.0);
        let debug_str = format!("{:?}", boundary_conditions);
        assert!(debug_str.contains("DirichletScalar"));

        // Debug string for Dirichlet vector boundary conditions.
        let boundary_conditions = BoundaryConditions1D::DirichletVector(
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        );
        let debug_str = format!("{:?}", boundary_conditions);
        assert!(debug_str.contains("DirichletVector"));

        // Debug string for Neumann scalar boundary conditions.
        let boundary_conditions = BoundaryConditions1D::NeumannScalar(0.0, 0.0);
        let debug_str = format!("{:?}", boundary_conditions);
        assert!(debug_str.contains("NeumannScalar"));

        // Debug string for Neumann vector boundary conditions.
        let boundary_conditions = BoundaryConditions1D::NeumannVector(
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        );
        let debug_str = format!("{:?}", boundary_conditions);
        assert!(debug_str.contains("NeumannVector"));
    }

    #[test]
    fn test_boundary_conditions_clone() {
        let boundary_conditions = BoundaryConditions1D::NeumannVector(
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
        );
        let cloned_boundary_conditions = boundary_conditions.clone();
        assert_eq!(boundary_conditions, cloned_boundary_conditions);
    }

    #[test]
    fn test_boundary_conditions_partial_eq() {
        let boundary_conditions_1 = BoundaryConditions1D::Periodic;
        let boundary_conditions_2 = BoundaryConditions1D::Periodic;
        let boundary_conditions_3 = BoundaryConditions1D::DirichletVector(
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
        );

        // Test equality with itself.
        assert_eq!(boundary_conditions_1, boundary_conditions_1);
        assert_eq!(boundary_conditions_1, boundary_conditions_2);

        // Test equality with a clone.
        assert_eq!(boundary_conditions_1, boundary_conditions_1.clone());

        // Test inequality.
        assert_ne!(boundary_conditions_1, boundary_conditions_3);
    }

    mod constructor_functions_tests {
        use super::*;

        #[test]
        fn test_new_none_bcs() {
            let boundary_conditions = BoundaryConditions1D::new_no_bcs();
            assert_eq!(boundary_conditions, BoundaryConditions1D::None);
        }

        #[test]
        fn test_new_periodic_bcs() {
            let boundary_conditions = BoundaryConditions1D::new_periodic_bcs();
            assert_eq!(boundary_conditions, BoundaryConditions1D::Periodic);
        }

        #[test]
        fn test_new_dirichlet_scalar_bcs() {
            let left_boundary_value = 0.0;
            let right_boundary_value = 1.0;
            let boundary_conditions =
                BoundaryConditions1D::new_dirichlet_scalar_bcs(
                    left_boundary_value,
                    right_boundary_value,
                );
            assert_eq!(
                boundary_conditions,
                BoundaryConditions1D::DirichletScalar(
                    left_boundary_value,
                    right_boundary_value
                )
            );
        }

        #[test]
        fn test_new_dirichlet_vector_bcs() {
            let left_boundary_value = [0.0, 0.0, 0.0];
            let right_boundary_value = [1.0, 1.0, 1.0];
            let boundary_conditions =
                BoundaryConditions1D::new_dirichlet_vector_bcs(
                    left_boundary_value,
                    right_boundary_value,
                );
            assert_eq!(
                boundary_conditions,
                BoundaryConditions1D::DirichletVector(
                    left_boundary_value,
                    right_boundary_value
                )
            );
        }

        #[test]
        fn test_new_neumann_scalar_bcs() {
            let left_boundary_value = 0.0;
            let right_boundary_value = 1.0;
            let boundary_conditions =
                BoundaryConditions1D::new_neumann_scalar_bcs(
                    left_boundary_value,
                    right_boundary_value,
                );
            assert_eq!(
                boundary_conditions,
                BoundaryConditions1D::NeumannScalar(
                    left_boundary_value,
                    right_boundary_value
                )
            );
        }

        #[test]
        fn test_new_neumann_vector_bcs() {
            let left_boundary_value = [0.0, 0.0, 0.0];
            let right_boundary_value = [1.0, 1.0, 1.0];
            let boundary_conditions =
                BoundaryConditions1D::new_neumann_vector_bcs(
                    left_boundary_value,
                    right_boundary_value,
                );
            assert_eq!(
                boundary_conditions,
                BoundaryConditions1D::NeumannVector(
                    left_boundary_value,
                    right_boundary_value
                )
            );
        }
    }
}
