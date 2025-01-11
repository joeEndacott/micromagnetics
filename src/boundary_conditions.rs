/// # Boundary conditions
///
/// ## Description
/// The `BoundaryConditions` enum represents the boundary conditions (BCs) that
/// can be applied to a scalar or vector field.
///
/// The BCs can be one of the following:
/// - `None`: No boundary conditions are applied.
/// - `Periodic`: Periodic boundary conditions are applied.
/// - `DirichletScalar`: Dirichlet boundary conditions are applied for a scalar
/// field. The boundary values are specified as an `f64` element.
/// - `DirichletVector`: Dirichlet boundary conditions are applied for a vector
/// field. The boundary values are specified as an of three `f64` elements.
/// - `NeumannScalar`: Neumann boundary conditions are applied for a scalar
/// field. The boundary values are specified as an `f64` element.
/// - `NeumannVector`: Neumann boundary conditions are applied for a vector
/// field. The boundary values are specified as an array of three `f64`
/// elements.
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
