from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt

from online_algorithm import OnlineAlgorithm
from sample_entropy import SampleEntropyAlgorithm
from shannon_entropy import ShannonEntropyAlgorithm
from renyi_entropy import RenyiEntropyAlgorithm
from permutation_entropy import PermutationEntropyAlgorithm


class WeightedCompositeEntropyAlgorithm(OnlineAlgorithm):
    
    def __init__(
        self,
        window_size: int = 100,
        threshold: float = 0.3,
        sample_entropy_params: Optional[Dict] = None,
        shannon_entropy_params: Optional[Dict] = None,
        renyi_entropy_params: Optional[Dict] = None,
        permutation_entropy_params: Optional[Dict] = None,
    ):
        self._window_size = window_size
        self._threshold = threshold
        
        self._entropy_algorithms = {
            'volatility': ShannonEntropyAlgorithm(
                window_size=window_size,
                **(shannon_entropy_params or {})
            ),
            'volume': SampleEntropyAlgorithm(
                window_size=window_size,
                **(sample_entropy_params or {})
            ),
            'financial_reports': PermutationEntropyAlgorithm(
                window_size=window_size,
                **(permutation_entropy_params or {})
            ),
            'macroeconomic': RenyiEntropyAlgorithm(
                window_size=window_size,
                **(renyi_entropy_params or {})
            )
        }
        
        self._buffers: Dict[str, deque] = {
            key: deque(maxlen=window_size * 2) 
            for key in self._entropy_algorithms.keys()
        }
        
        self._entropy_values: Dict[str, List[float]] = {
            key: [] for key in self._entropy_algorithms.keys()
        }
        
        self._composite_entropy_values: List[float] = []
        self._position: int = 0
        self._last_change_point: Optional[int] = None
        
    def detect(self, observation: Dict[str, np.float64] | Dict[str, npt.NDArray[np.float64]]) -> bool:
        if not isinstance(observation, dict):
            raise ValueError("Наблюдение должно быть словарем, сопоставляющим названия факторов со значениями")
            
        self._process_observations(observation)
        return self._last_change_point is not None
        
    def localize(self, observation: Dict[str, np.float64] | Dict[str, npt.NDArray[np.float64]]) -> Optional[int]:
        change_detected = self.detect(observation)
        
        if change_detected:
            change_point = self._last_change_point
            self._last_change_point = None
            return change_point
            
        return None
        
    def _process_observations(self, observations: Dict[str, np.float64 | npt.NDArray[np.float64]]) -> None:
        self._position += 1
        
        for factor, value in observations.items():
            if factor in self._buffers:
                if isinstance(value, np.ndarray):
                    self._buffers[factor].append(float(value[0]))
                else:
                    self._buffers[factor].append(float(value))
        
        if all(len(buffer) >= self._window_size for buffer in self._buffers.values()):
            entropy_results = {}
            
            for factor, algorithm in self._entropy_algorithms.items():
                if factor in self._buffers and len(self._buffers[factor]) >= self._window_size:
                    window = np.array(list(self._buffers[factor])[-self._window_size:])
                    
                    if isinstance(algorithm, SampleEntropyAlgorithm):
                        entropy = algorithm._calculate_sample_entropy(window)
                    elif isinstance(algorithm, ShannonEntropyAlgorithm):
                        hist, _ = np.histogram(window, bins=algorithm._bins, density=True)
                        hist = hist / np.sum(hist)
                        entropy = algorithm._compute_entropy(hist)
                    elif isinstance(algorithm, RenyiEntropyAlgorithm):
                        entropy = algorithm._calculate_renyi_entropy(window)
                    elif isinstance(algorithm, PermutationEntropyAlgorithm):
                        entropy = algorithm._calculate_permutation_entropy(window)
                    else:
                        entropy = 0.0
                    
                    if np.isinf(entropy) or np.isnan(entropy):
                        entropy = 0.0
                        
                    entropy_results[factor] = entropy
                    self._entropy_values[factor].append(entropy)
            
            if len(entropy_results) > 0:
                composite_entropy = self._calculate_weighted_entropy(list(entropy_results.values()))
                self._composite_entropy_values.append(composite_entropy)
                
                if len(self._composite_entropy_values) >= 2:
                    entropy_diff = abs(
                        self._composite_entropy_values[-1] - 
                        self._composite_entropy_values[-2]
                    )
                    
                    if entropy_diff > self._threshold:
                        self._last_change_point = self._position - self._window_size // 2
                        
    def _calculate_weighted_entropy(self, entropy_values: List[float]) -> float:
        if not entropy_values:
            return 0.0
            
        c_values = np.array(entropy_values)
        
        exp_values = np.exp(c_values)
        
        p_sum = np.sum(exp_values)
        if p_sum == 0:
            return 0.0
            
        p_weights = exp_values / p_sum
        
        weighted_entropy = np.sum(c_values * p_weights)
        
        return float(weighted_entropy)
        
    def get_entropy_history(self) -> Dict[str, List[float]]:
        return {
            key: values.copy() 
            for key, values in self._entropy_values.items()
        }
        
    def get_composite_entropy_history(self) -> List[float]:
        return self._composite_entropy_values.copy()
        
    def get_current_weights(self) -> Optional[Dict[str, float]]:
        if not all(self._entropy_values[key] for key in self._entropy_algorithms.keys()):
            return None
            
        latest_entropies = {
            key: values[-1] if values else 0.0
            for key, values in self._entropy_values.items()
        }
        
        c_values = np.array(list(latest_entropies.values()))
        exp_values = np.exp(c_values)
        p_sum = np.sum(exp_values)
        
        if p_sum == 0:
            return None
            
        p_weights = exp_values / p_sum
        
        return {
            key: float(p_weights[i])
            for i, key in enumerate(latest_entropies.keys())
        }
        
    def reset(self) -> None:
        for buffer in self._buffers.values():
            buffer.clear()
        for values in self._entropy_values.values():
            values.clear()
        self._composite_entropy_values.clear()
        self._position = 0
        self._last_change_point = None
        
        for algorithm in self._entropy_algorithms.values():
            if hasattr(algorithm, 'reset'):
                algorithm.reset()


if __name__ == "__main__":
    algorithm = WeightedCompositeEntropyAlgorithm(
        window_size=40,
        threshold=0.2,
        sample_entropy_params={'m': 2, 'r_factor': 0.2},
        shannon_entropy_params={'bins': 10},
        renyi_entropy_params={'alpha': 0.5},
        permutation_entropy_params={'embedding_dimension': 3}
    )
    
    np.random.seed(42)
    n_samples = 200
    
    normal_data = {
        'volatility': np.random.normal(0, 1, n_samples),
        'volume': np.random.exponential(1, n_samples),
        'financial_reports': np.random.uniform(0, 1, n_samples),
        'macroeconomic': np.random.normal(0, 0.5, n_samples)
    }
    
    for i in range(n_samples):
        observation = {
            key: data[i] for key, data in normal_data.items()
        }
        
        change_detected = algorithm.detect(observation)
        
        if change_detected:
            change_point = algorithm.localize(observation)
            print(f"Точка изменения обнаружена на позиции: {change_point}")
    
    weights = algorithm.get_current_weights()
    if weights:
        print("\nФинальные веса softmax:")
        for factor, weight in weights.items():
            print(f"{factor}: {weight:.4f}")
    
    composite_history = algorithm.get_composite_entropy_history()
    print(f"\nЗначения композитной энтропии (последние 5): {composite_history[-5:]}")