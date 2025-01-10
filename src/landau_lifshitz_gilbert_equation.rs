use crate::vector_field::VectorField1D;

/// # Get effective field
///
/// ## Description
/// Returns the effective field given the current magnetization.
///
/// ## Example use case
/// ```
/// use grid::Grid;
/// use vector_field::VectorField1D;
///
/// let grid = Grid::new_uniform_grid(0.0, 1.0, 100);
/// let exchange_constant = 1.0;
/// let dmi_constant = 1.0;
/// let magnetization = VectorField1D::new_constant_vector_field(&grid, [0.0, 0.0, 1.0]);
/// let effective_field = get_effective_field(exchange_constant, dmi_constant, &magnetization);
/// ```
///
fn get_effective_field(
    exchange_constant: f64,
    dmi_constant: f64,
    magnetization: &VectorField1D,
) -> VectorField1D {
    exchange_effective_field(exchange_constant, magnetization)
        .add(&dmi_effective_field(dmi_constant, magnetization))
        .unwrap()
}

/// # Exchange effective field
///
/// ## Description
/// Returns the exchange effective field given the current magnetization.
///
/// ## Example use case
/// ```
/// use grid::Grid;
/// use vector_field::VectorField1D;
///
/// let grid = Grid::new_uniform_grid(0.0, 1.0, 100);
/// let exchange_constant = 1.0;
/// let magnetization = VectorField1D::new_constant_vector_field(&grid, [0.0, 0.0, 1.0]);
/// let exchange_effective_field = exchange_effective_field(exchange_constant, &magnetization);
/// ```
///
fn exchange_effective_field(
    exchange_constant: f64,
    magnetization: &VectorField1D,
) -> VectorField1D {
    magnetization.laplacian().scale(exchange_constant)
}

/// # DMI effective field
///
/// ## Description
/// Returns the DMI effective field given the current magnetization.
///
/// ## Example use case
/// ```
/// use grid::Grid;
/// use vector_field::VectorField1D;
///
/// let grid = Grid::new_uniform_grid(0.0, 1.0, 100);
/// let dmi_constant = 1.0;
/// let magnetization = VectorField1D::new_constant_vector_field(&grid, [0.0, 0.0, 1.0]);
/// let dmi_effective_field = dmi_effective_field(dmi_constant, &magnetization);
/// ```
///
fn dmi_effective_field(
    dmi_constant: f64,
    magnetization: &VectorField1D,
) -> VectorField1D {
    magnetization.curl().scale(-1.0 * dmi_constant)
}

/// # Landau-Lifshitz-Gilbert right-hand side
///
/// ## Description
/// Returns the right-hand side of the Landau-Lifshitz-Gilbert equation in the
/// micromagnetic limit, given  the exchange constant, DMI constant, damping
/// constant, gyromagnetic ratio, and current magnetization.
///
/// `alpha` is the damping constant, `gamma` is the gyromagnetic ratio, and
/// `magnetization` is the current magnetization.
///
/// ## Example use case
/// ```
/// use grid::Grid;
/// use vector_field::VectorField1D;
///
/// let grid = Grid::new_uniform_grid(0.0, 1.0, 100);
/// let exchange_constant = 1.0;
/// let dmi_constant = 1.0;
/// let alpha = 0.1;
/// let gamma = 1.0;
/// let magnetization = VectorField1D::new_constant_vector_field(&grid, [0.0, 0.0, 1.0]);
/// let llg_rhs = landau_lifshitz_gilbert_rhs(exchange_constant, dmi_constant, alpha, gamma, &magnetization);
/// ```
///
pub fn landau_lifshitz_gilbert_rhs(
    exchange_constant: f64,
    dmi_constant: f64,
    alpha: f64,
    gamma: f64,
    magnetization: &VectorField1D,
) -> VectorField1D {
    let effective_field =
        get_effective_field(exchange_constant, dmi_constant, magnetization);

    magnetization
        .cross_product(&effective_field)
        .unwrap()
        .scale(alpha)
        .add(&effective_field)
        .unwrap()
        .cross_product(magnetization)
        .unwrap()
        .scale(gamma / (1.0 + alpha.powi(2)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::Grid;

    #[test]
    fn test_get_effective_field() {
        let grid = Grid::new_uniform_grid(0.0, 1.0, 100);
        let exchange_constant = 1.0;
        let dmi_constant = 1.0;
        let magnetization =
            VectorField1D::new_constant_vector_field(&grid, [0.0, 0.0, 1.0]);
        let effective_field = get_effective_field(
            exchange_constant,
            dmi_constant,
            &magnetization,
        );

        // Verify that the grids of the two vectors are equal
        assert_eq!(effective_field.grid, magnetization.grid);

        // Verify that the two vector fields are compatible
        assert_eq!(effective_field.scale(0.0), magnetization.scale(0.0));
    }

    #[test]
    fn test_exchange_effective_field() {
        let grid = Grid::new_uniform_grid(0.0, 1.0, 100);
        let exchange_constant = 1.0;
        let magnetization =
            VectorField1D::new_constant_vector_field(&grid, [0.0, 0.0, 1.0]);
        let exchange_field =
            exchange_effective_field(exchange_constant, &magnetization);

        // Verify that the grids of the two vectors are equal
        assert_eq!(exchange_field.grid, magnetization.grid);

        // Verify that the two vector fields are compatible
        assert_eq!(exchange_field.scale(0.0), magnetization.scale(0.0));
    }

    #[test]
    fn test_dmi_effective_field() {
        let grid = Grid::new_uniform_grid(0.0, 1.0, 100);
        let dmi_constant = 1.0;
        let magnetization =
            VectorField1D::new_constant_vector_field(&grid, [0.0, 0.0, 1.0]);
        let dmi_field = dmi_effective_field(dmi_constant, &magnetization);

        // Verify that the grids of the two vectors are equal
        assert_eq!(dmi_field.grid, magnetization.grid);

        // Verify that the two vector fields are compatible
        assert_eq!(dmi_field.scale(0.0), magnetization.scale(0.0));
    }

    #[test]
    fn test_landau_lifshitz_gilbert_rhs() {
        let grid = Grid::new_uniform_grid(0.0, 1.0, 100);
        let exchange_constant = 1.0;
        let dmi_constant = 1.0;
        let alpha = 0.1;
        let gamma = 1.0;
        let magnetization =
            VectorField1D::new_constant_vector_field(&grid, [0.0, 0.0, 1.0]);
        let llg_rhs = landau_lifshitz_gilbert_rhs(
            exchange_constant,
            dmi_constant,
            alpha,
            gamma,
            &magnetization,
        );

        // Verify that the grids of the two vectors are equal
        assert_eq!(llg_rhs.grid, magnetization.grid);

        // Verify that the two vector fields are compatible
        assert_eq!(llg_rhs.scale(0.0), magnetization.scale(0.0));
    }
}
