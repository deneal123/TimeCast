"""Общие фиче-хелперы (без torch): retail-дефолт, список фич, число каналов.

Единственный источник правды о том, какие фичи у ряда и сколько каналов у модели —
раньше это было зашито в data/neiro, training/neiro, inference/neiro и prepare.
"""

# Доменные retail-фичи по умолчанию (когда dictidx без 'feature_cols').
RETAIL_FEATURES = ["sell_price", "event_name", "event_type", "cashback"]


def feature_cols(dictidx) -> list:
    """Список фич ряда: обобщённый формат задаёт 'feature_cols', иначе retail-дефолт."""
    return list(dictidx.get("feature_cols") or RETAIL_FEATURES)


def num_variates(dictidx) -> int:
    """Число каналов модели: 3 (decomposition resid/trend/season) + число фич.

    Обучение всегда в режиме декомпозиции (collate_fn передаёт scalers), поэтому
    значение не зависит от домена (retail: 3 + 4 = 7) и не требуется в запросе.
    """
    return 3 + len(feature_cols(dictidx))
