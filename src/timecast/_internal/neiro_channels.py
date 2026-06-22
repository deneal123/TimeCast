"""Сборка входного тензора neiro-модели из батча (torch).

Убирает дублирование блоков torch.cat между training/neiro и inference/neiro.
Фиче-логика (число каналов, список фич) — в features.py (без torch).
"""
import torch


def is_process_batch(batch) -> bool:
    """Режим декомпозиции активен, если collate_fn положил resid/trend/season."""
    return len(batch['train']) > 4


def assemble_input(part, proccess: bool, device) -> torch.Tensor:
    """Собирает входной тензор модели [batch, seq_len, channels] из части батча.

    part — batch['train'] или batch['test']. В режиме декомпозиции каналы =
    resid/trend/season + фичи; иначе сам ряд + фичи.
    """
    exogenous = part["exogenous"].to(device)
    if proccess:
        resid = part["resid"].to(device).unsqueeze(-1)
        trend = part["trend"].to(device).unsqueeze(-1)
        season = part["season"].to(device).unsqueeze(-1)
        return torch.cat((resid, trend, season, exogenous), dim=-1)
    series = part["series"].to(device).unsqueeze(-1)
    return torch.cat((series, exogenous), dim=-1)
