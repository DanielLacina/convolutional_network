use crate::filter::{Filter, Padding}; 
use crate::activation::Activation;
use std::iter::zip;
use crate::utils::relu;


pub struct Layer {
    filters: Vec<Filter>,
    num_filters: i32,
    activation: Activation,
    padding: Padding,
    input_shape: (i32, i32, i32),
    strides: (i32, i32),
    kernal_size: (i32, i32)
}

impl Layer {
    pub fn new(num_filters: i32,  kernal_size: (i32, i32), strides: (i32, i32), activation: Activation, padding: Padding, input_shape: (i32, i32, i32)) -> Self {
       let filters = (0..num_filters).map(|_| Filter::new(kernal_size, strides, padding.clone())).collect();
       Self {
        filters,
        num_filters,
        activation,
        padding,
        input_shape,
        strides,
        kernal_size
       }
    }

    pub fn apply_filters(&self, matrix: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
        let output_matrices: Vec<Vec<Vec<f32>>> = self.filters.iter().map(|filter| filter.apply_filter(matrix)).collect(); 
        let mut output_matrix = output_matrices.get(0).unwrap().clone();
        for matrix in output_matrices[1..].iter() {
            output_matrix = zip(output_matrix, matrix).map(|(v1, v2)| zip(v1, v2).map(|(c1, c2)| c1 + *c2).collect()).collect();
        }
        self.apply_activation(&mut output_matrix, self.activation.clone());
        return output_matrix;
    }

    fn apply_activation(&self, matrix: &mut Vec<Vec<f32>>, activation: Activation) {
        match activation {
            Activation::Relu => {
                *matrix = matrix.iter().map(|v| v.iter().map(|c| relu(*c)).collect()).collect();      
            }
            Activation::None => {} 
        } 
    } 
}

#[cfg(test)]
mod tests {
    use super::*;

   #[test]
   fn test_layer_initialization() {
       let num_filters = 3;
       let activation = Activation::Relu;
       let padding = Padding::Zero;
       let input_shape = (3, 3, 1);
       let strides = (1, 1);
       let kernal_size = (1, 1);
       let layer = Layer::new(num_filters, kernal_size, strides, activation, padding, input_shape);
       assert!(matches!(layer.activation, Activation::Relu));
       assert!(matches!(layer.padding, Padding::Zero));
       assert!(layer.input_shape == input_shape);
       assert!(layer.strides == strides);
       assert!(layer.kernal_size == kernal_size);
       assert!(layer.filters.len() == 3);
   }

   #[test]
   fn test_apply_filter() {
       let num_filters = 3; 
       let activation = Activation::Relu;
       let padding = Padding::Zero;
       let input_shape = (50, 50, 1);
       let strides = (2, 2);
       let kernal_size = (1, 1);
       let layer = Layer::new(num_filters, kernal_size, strides, activation, padding, input_shape);
       let matrix: Vec<Vec<f32>> = (0..input_shape.0)
            .map(|i| {
                (0..input_shape.1)
                    .map(|j| ((i + j) % 255 + 1) as f32)
                    .collect()
            })
            .collect();
       let output_matrix = layer.apply_filters(&matrix);
       assert!(output_matrix.len() as i32 == matrix.len() as i32/strides.0);
       assert!(output_matrix[0].len() as i32 == matrix[0].len() as i32/strides.1);
       assert!(output_matrix.iter().all(|v| v.iter().all(|c| *c > 0.0)));
   }
}