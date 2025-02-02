use crate::errors::FieldError;
use crate::scalar_field::ScalarField1D;
use crate::vector_field::VectorField1D;

pub trait Field {
    type Grid;
    type FieldValues;
    type BoundaryConditions;

    fn grid(&self) -> Self::Grid;
    fn field_values(&self) -> Self::FieldValues;
    fn boundary_conditions(&self) -> Self::BoundaryConditions;
    fn new(grid: Self::Grid, field_values: Self::FieldValues) -> Self;
}

pub trait FieldArithmetic {
    fn add(&self, _other: &Self) -> Result<Self, FieldError>
    where
        Self: Sized,
    {
        if self.grid != other.grid {
            return Err(FieldError::GridMismatch);
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
            boundary_conditions: BoundaryConditions1D::default(),
        })
    }

    fn subtract(&self, _other: &Self) -> Result<Self, FieldError>
    where
        Self: Sized,
    {
        Err(FieldError::OperationNotSupported)
    }

    fn multiply(&self, _other: &Self) -> Result<Self, FieldError>
    where
        Self: Sized,
    {
        Err(FieldError::OperationNotSupported)
    }

    fn divide(&self, _other: &Self) -> Result<Self, FieldError>
    where
        Self: Sized,
    {
        Err(FieldError::OperationNotSupported)
    }

    fn scale(&self, _scalar: f64) -> Result<Self, FieldError>
    where
        Self: Sized,
    {
        Err(FieldError::OperationNotSupported)
    }
}

pub trait Calculus {
    fn partial_x(&self) -> Self;
    fn partial_y(&self) -> Self;
    fn partial_z(&self) -> Self;
    fn gradient(&self) -> VectorField1D;
    fn divergence(&self) -> ScalarField1D;
    fn laplacian(&self) -> Self;
    fn curl(&self) -> VectorField1D;
}

pub trait BoundaryConditions {
    fn check_boundary_conditions(&self) -> bool;
    fn apply_boundary_conditions(&self) -> Self;
}
