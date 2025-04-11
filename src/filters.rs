use rand::Rng;
use std::collections::VecDeque;
use std::iter::zip;

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
   kernal_size: (i32, i32),  
   stride: (i32, i32),
   padding: Padding,
   weights: Vec<Vec<f32>>
} 

impl Filter {
    pub fn new(kernal_size: (i32, i32), stride: (i32, i32), padding: Padding) -> Self {
        let weights = Filter::initialize_weights(kernal_size) ;
        return Self {
            weights,
            kernal_size,
            stride,
            padding
        }
    }

    fn initialize_weights(kernal_size: (i32, i32)) -> Vec<Vec<f32>> {
        let mut rng = rand::rng();
        let weights = (0..kernal_size.0).map(|_| (0..kernal_size.1).map(|_| rng.random::<f32>()).collect()).collect();   
        return weights;
    }


    fn compute_convolution(&self, matrix: &Vec<Vec<f32>>, m_i: i32, v_i: i32) -> f32 {
         let weights_len = (self.weights.len() + self.weights.get(0).unwrap().len()) as f32; 
         let matrix_slice = self.matrix_slice(matrix, m_i, v_i);
         let sum: f32 = zip(matrix_slice, self.weights.iter()).fold(0.0, |sum_m, (v1, v2)| 
              sum_m + zip(v1, v2).fold(0.0, |sum_v,  (c1, c2)| sum_v + (c1 * c2)));
         return sum/weights_len;
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

    fn matrix_slice(&self, matrix: &Vec<Vec<f32>>, m_i: i32, v_i: i32) -> VecDeque<VecDeque<f32>> {
        let vertical_min = m_i - (self.kernal_size.0/2);
        let vertical_max = m_i  + (self.kernal_size.0/2);
        let horizontal_min = v_i - (self.kernal_size.1/2);
        let horizontal_max = v_i  + (self.kernal_size.1/2);
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
            self.add_vertical_padding(&mut matrix_slice, VerticalDirection::Top, underflow, self.kernal_size.1);
        }
        if vertical_max > matrix.len() as i32 - 1 {
            let overflow = vertical_max - matrix.len() as i32 - 1; 
            self.add_vertical_padding(&mut matrix_slice, VerticalDirection::Bottom, overflow, self.kernal_size.1);
        }
        return matrix_slice;
   }   

    pub fn apply_filter(&self, matrix: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
        let mut result_matrix = Vec::new();
        for m_i in (0..matrix.len()) {
             if m_i as i32 % self.stride.0 != 0 {
                 continue;
             }
             let mut vector = Vec::new();
             for v_i in (0..matrix.get(m_i).unwrap().len())  {
                if v_i as i32 % self.stride.1 != 0 {
                   continue;
                }
                let convolution = self.compute_convolution(matrix, m_i as i32, v_i as i32);
                vector.push(convolution);
             } 
             result_matrix.push(vector);
        }
        return result_matrix;
}

}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_filter_initialization() {
         let kernal_size = (3, 3);
         let stride = (1, 1);
         let padding = Padding::Ones; 
         let filter = Filter::new(kernal_size, stride, padding); 
         assert!(filter.kernal_size == kernal_size);
         assert!(matches!(filter.padding, Padding::Ones));
         assert!(filter.stride == (1, 1));
         assert!(filter.weights.len() as i32 == kernal_size.0);
         assert!(filter.weights.iter().all(|v| v.len() as i32 == kernal_size.1));
         assert!(filter.weights.iter().all(|v| v.iter().all(|c| *c <= 1.0 && *c > 0.0)));
    } 
}

