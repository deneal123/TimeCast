import os
import torch
import torch.nn as nn




class CustomLoss(torch.nn.Module):
    def __init__(self, beta=1.0, delta=0.5, gamma=0.1, cosine_margin=0.0, special_penalty=1.0):
        """
        Инициализация кастомной функции потерь.

        Параметры:
        - beta: параметр для SmoothL1Loss.
        - delta: вес для штрафа за отрицательные значения.
        - gamma: вес для CosineEmbeddingLoss.
        - cosine_margin: маржа для CosineEmbeddingLoss.
        - special_penalty: штраф для второй и последующих компонент.
        """
        super(CustomLoss, self).__init__()
        self.smooth_l1_loss = torch.nn.SmoothL1Loss(beta=beta)
        self.cosine_loss = torch.nn.CosineEmbeddingLoss(margin=cosine_margin)
        self.delta = delta
        self.gamma = gamma
        self.special_penalty = special_penalty  # Специальный штраф для второй и последующих компонент

    def forward(self, pred, target, aux_input=None):
        """
        Вычисление функции потерь.

        Параметры:
        - pred: предсказания модели с формой (batch_size, seq_len, num_components).
        - target: целевые значения с формой (batch_size, seq_len, num_components).
        - aux_input: вспомогательные входные данные для вычисления CosineEmbeddingLoss.

        Возвращает:
        - loss: итоговая функция потерь.
        """
        num_components = pred.shape[-1]
        total_loss = 0

        for i in range(num_components):
            # Берем отдельную компоненту
            pred_component = pred[..., i]
            target_component = target[..., i]

            # SmoothL1Loss
            l1_loss = self.smooth_l1_loss(pred_component, target_component)

            # Штраф за отрицательные значения
            penalty = torch.mean(torch.clamp(-pred_component, min=0) ** 2)  # Квадрат штрафа за отрицательные предсказания

            # CosineEmbeddingLoss
            if aux_input is not None:
                aux_component = aux_input[..., i]
                cosine_labels = torch.ones(pred_component.size(0)).to(pred.device)
                cosine_loss = self.cosine_loss(pred_component, aux_component, cosine_labels)
            else:
                cosine_loss = 0.0

            # Специальный штраф для второй компоненты
            if i > 1:
                special_penalty = self.special_penalty
            else:
                special_penalty = 0.0
                
            # Итоговая функция потерь для компоненты с учетом специального штрафа
            component_loss = l1_loss + self.delta * penalty + self.gamma * cosine_loss + special_penalty
            total_loss += component_loss

        return total_loss