from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd

# Import the main explanation function from predict.py
# Rename it to avoid naming conflict with this wrapper function
from ml_pipeline.predict import explain_prediction as _explain_prediction


def explain_prediction(
    input_df: pd.DataFrame,
    class_index: Optional[int] = None,
    top_k: int = 5,
) -> Dict[str, Any]:
    # Validate input before sending to the main explanation function
    if input_df is None or not isinstance(input_df, pd.DataFrame) or input_df.empty:
        raise ValueError("input_df must be a non-empty pandas DataFrame.")

    # top_k means how many most important features to return
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0.")

    # Reuse the explanation logic already written in predict.py
    return _explain_prediction(
        input_df=input_df,
        class_index=class_index,
        top_k=top_k,
    )