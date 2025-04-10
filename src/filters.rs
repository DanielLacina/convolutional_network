use crate::matrices::dot_matrices; 
use rand::Rng;
use std::collections::VecDeque;

enum Padding {
   Zero,
   Ones, 
   Reflective,
}

enum HorizontalDirection {
    Front,
    Back
}

enum VerticalDirection {
    Top, 
    Bottom
}

pub struct Filter {
   kernal_size: i32,  
   stride: i32,
   padding: Padding,
   weights: Vec<Vec<f32>>
} 

impl Filter {
    pub fn new(kernal_size: i32, stride: i32, padding: Padding) -> Self {
        let weights = Filter::initialize_weights(kernal_size) ;
        return Self {
            weights,
            kernal_size,
            stride,
            padding
        }
    }

    fn initialize_weights(kernal_size: i32) -> Vec<Vec<f32>> {
        let mut rng = rand::rng();
        let weights = (0..kernal_size).map(|_| (0..kernal_size).map(|_| rng.random::<f32>()).collect()).collect();   
        return weights;
    }


    fn add_vertical_padding(&self, matrix_slice: &mut VecDeque<VecDeque<f32>>, padding_direction: VerticalDirection, m_amount: i32, v_amount: i32) {
         match padding_direction {
            VerticalDirection::Bottom => {
                for _ in (0..m_amount) {
                    let mut vector = VecDeque::new();
                    self.add_horizontal_padding(&mut vector , HorizontalDirection::Back, v_amount);
                    matrix_slice.push_back(vector);
                }
            } 
            VerticalDirection::Top => {
                for _ in (0..m_amount) {
                    let mut vector = VecDeque::new();
                    self.add_horizontal_padding(&mut vector , HorizontalDirection::Back, v_amount);
                    matrix_slice.push_front(vector);
                }
            }
         }   
    }

    fn add_horizontal_padding(&self, vector_slice: &mut VecDeque<f32>, padding_direction: HorizontalDirection, amount: i32) {
         match padding_direction {
            HorizontalDirection::Back => {
                for _ in (0..amount) { 
                    vector_slice.push_back(0.0);
                }
            } 
            HorizontalDirection::Front => {
                for _ in (0..amount) { 
                    vector_slice.push_front(0.0);
                }
            }
         }   
    }

    fn matrix_slice(&self, matrix: &Vec<Vec<f32>>, i: i32, j: i32) -> VecDeque<VecDeque<f32>> {
        let vertical_min = i - (self.kernal_size/2);
        let vertical_max = i  + (self.kernal_size/2);
        let horizontal_min = j - (self.kernal_size/2);
        let horizontal_max = j  + (self.kernal_size/2);
        let mut matrix_slice = VecDeque::new();
        for k in (vertical_min.max(0)..vertical_max.min(matrix.len() as i32 - 1) + 1) {
            let vector = matrix.get(k as usize).unwrap();
            let mut vector_slice = VecDeque::new();
            for h in (horizontal_min.max(0)..horizontal_max.min(vector.len() as i32 - 1) + 1)  {
                    vector_slice.push_back(vector[h as usize].clone()) ;  
            }
            if horizontal_min < 0 { 
                let underflow = - horizontal_min; 
                self.add_horizontal_padding(&mut vector_slice, HorizontalDirection::Front, underflow);
            }
            if horizontal_max > vector.len() as i32 - 1{ 
                let overflow = horizontal_max - vector.len() as i32 - 1;
                self.add_horizontal_padding(&mut vector_slice, HorizontalDirection::Back, overflow);
            }
            matrix_slice.push_back(vector_slice);
        } 
        if vertical_min < 0 {
            let underflow = - vertical_min; 
            self.add_vertical_padding(&mut matrix_slice, VerticalDirection::Top, underflow, self.kernal_size);
        }
        if vertical_max > matrix.len() as i32 - 1 {
            let overflow = vertical_max - matrix.len() as i32 - 1; 
            self.add_vertical_padding(&mut matrix_slice, VerticalDirection::Bottom, overflow, self.kernal_size);
        }
        return matrix_slice;
   }   

    pub fn apply_filter(&self, matrix: &Vec<Vec<f32>>) {
        let mut matrix = matrix.clone();
        for i in (0..matrix.len()) {
             if i as i32 % self.stride != 0 {
                 continue;
             }
             for j in (0..matrix.get(i).unwrap().len())  {
                if j as i32 % self.stride != 0 {
                   continue;
                }
                let matrix_slice = self.matrix_slice(&matrix, i as i32, j as i32);

             } 
    }
}

}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_filter_initialization() {
         let kernal_size = 3 ;
         let stride = 1;
         let padding = Padding::Ones; 
         let filter = Filter::new(kernal_size, stride, padding); 
         assert!(filter.kernal_size == kernal_size);
         assert!(matches!(filter.padding, Padding::Ones));
         assert!(filter.stride == 1);
         assert!(filter.weights.len() as i32 == kernal_size);
         assert!(filter.weights.iter().all(|v| v.len() as i32 == kernal_size));
         assert!(filter.weights.iter().all(|v| v.iter().all(|c| *c <= 1.0 && *c > 0.0)));
    } 
    #[test]
    fn test_apply_filter() {
         let kernal_size = 10;
         let stride = 1;
         let padding = Padding::Ones; 
         let filter = Filter::new(kernal_size, stride, padding); 
         let matrix: Vec<Vec<f32>> = vec![vec![1.0, 2.0, 3.0, 4.0], vec![1.0, 3.0, 5.0, 6.0], vec![3.0, 9.0, 4.0, 2.0]];
         println!("{:?}", filter.apply_filter(&matrix));

    } 
}

