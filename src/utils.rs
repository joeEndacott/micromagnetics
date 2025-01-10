use crate::scalar_field::ScalarField1D;
use crate::vector_field::VectorField1D;

/// # Test scalar equality with tolerance
///
/// ## Description
/// Tests if two scalar are equal to within a given tolerance.
///
pub fn scalar_equality(num_1: f64, num_2: f64, tolerance: f64) -> bool {
    (num_1 - num_2).abs() < tolerance
}

/// # Test vector equality with tolerance
///
/// ## Description
/// Tests if two 3-vectors are equal to within a given tolerance.
///
pub fn vector_equality(
    vec_1: [f64; 3],
    vec_2: [f64; 3],
    tolerance: f64,
) -> bool {
    (vec_1[0] - vec_2[0]).abs() < tolerance
        && (vec_1[1] - vec_2[1]).abs() < tolerance
        && (vec_1[2] - vec_2[2]).abs() < tolerance
}

/// # Test scalar field equality with tolerance
///
/// ## Description
/// Tests if two scalar fields are equal within a given tolerance.
///
pub fn scalar_field_equality(
    scalar_field_1: &ScalarField1D,
    scalar_field_2: &ScalarField1D,
    tolerance: f64,
) -> bool {
    if scalar_field_1.grid != scalar_field_2.grid {
        return false;
    }

    scalar_field_1
        .field_values
        .iter()
        .zip(scalar_field_2.field_values.iter())
        .all(|(v1, v2)| scalar_equality(*v1, *v2, tolerance))
}

/// # Test vector field equality with tolerance
///
/// ## Description
/// Tests if two vector fields are equal within a given tolerance.
///
pub fn vector_field_equality(
    vector_field_1: &VectorField1D,
    vector_field_2: &VectorField1D,
    tolerance: f64,
) -> bool {
    if vector_field_1.grid != vector_field_2.grid {
        return false;
    }

    vector_field_1
        .field_values
        .iter()
        .zip(vector_field_2.field_values.iter())
        .all(|(v1, v2)| vector_equality(*v1, *v2, tolerance))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::boundary_conditions::BoundaryConditions1D;
    use crate::grid::Grid;
    use crate::vector_field::VectorField1D;

    #[test]
    fn test_scalar_equality_within_tolerance() {
        assert!(scalar_equality(1.0, 1.0001, 0.001));
        assert!(scalar_equality(1.0, 0.9999, 0.001));
        assert!(!scalar_equality(1.0, 1.01, 0.001));
        assert!(!scalar_equality(1.0, 0.99, 0.001));
    }

    #[test]
    fn test_vector_equality_within_tolerance() {
        assert!(vector_equality(
            [1.0, 2.0, 3.0],
            [1.0001, 2.0001, 2.9999],
            0.001
        ));
        assert!(!vector_equality([1.0, 2.0, 3.0], [1.01, 2.01, 2.99], 0.001));
    }

    #[test]
    fn test_scalar_field_equality_within_tolerance() {
        let grid = Grid::new_uniform_grid(0.0, 1.0, 10);
        let scalar_field_1 = ScalarField1D::new_constant_scalar_field(
            &grid,
            1.0,
            &BoundaryConditions1D::None,
        )
        .unwrap();
        let scalar_field_2 = ScalarField1D::new_constant_scalar_field(
            &grid,
            1.0001,
            &BoundaryConditions1D::None,
        )
        .unwrap();

        assert!(scalar_field_equality(
            &scalar_field_1,
            &scalar_field_2,
            0.001
        ));

        let scalar_field_2 = ScalarField1D::new_constant_scalar_field(
            &grid,
            1.01,
            &BoundaryConditions1D::None,
        )
        .unwrap();
        assert!(!scalar_field_equality(
            &scalar_field_1,
            &scalar_field_2,
            0.001
        ));
    }

    #[test]
    fn test_vector_field_equality_within_tolerance() {
        let grid = Grid::new_uniform_grid(0.0, 1.0, 10);
        let vector_field_1 = VectorField1D::new_constant_vector_field(
            &grid,
            [1.0, 1.0, 1.0],
            &BoundaryConditions1D::None,
        )
        .unwrap();
        let vector_field_2 = VectorField1D::new_constant_vector_field(
            &grid,
            [1.0001, 1.0001, 0.9999],
            &BoundaryConditions1D::None,
        )
        .unwrap();

        assert!(vector_field_equality(
            &vector_field_1,
            &vector_field_2,
            0.001
        ));

        let vector_field_2 = VectorField1D::new_constant_vector_field(
            &grid,
            [1.01, 1.01, 0.99],
            &BoundaryConditions1D::None,
        )
        .unwrap();
        assert!(!vector_field_equality(
            &vector_field_1,
            &vector_field_2,
            0.001
        ));
    }
}
