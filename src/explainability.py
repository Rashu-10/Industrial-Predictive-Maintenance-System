import numpy as np

class ModelExplainer:
  def __init__(self, model):
    self.model = model

  def explain(self, scaled_input):
    """
    Calculate local feature impact using deviation-weighted importance.
    This replaces the heavy 'shap' library dependency to ensure instant loading
    and deployment stability on Streamlit Community Cloud.
    """
    # Importances of features globally
    importances = self.model.feature_importances_
    
    # Calculate local impact: absolute standardized deviation * global feature importance
    # scaled_input[0] contains the standardized deviations (Z-scores) of the features.
    local_impacts = np.abs(scaled_input[0]) * importances
    
    # Reshape to match the SHAP output shape expected by prediction.py: (1, n_features, 2)
    # where the third axis represents binary class contributions (index 1 for failure contribution)
    mock_shap = np.zeros((1, len(importances), 2))
    mock_shap[0, :, 1] = local_impacts
    
    return mock_shap