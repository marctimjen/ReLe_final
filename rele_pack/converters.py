import torch
def action_to_hot(action):
    if action == (0, 0):  # Move up and left
        return 0
    elif action == (0, 1):  # Move up
        return 1
    elif action == (1, 0):  # Move left
        return 2
    elif action == (1, 1):  # stand still
        return 3
    elif action == (0, 2):  # Move up and right
        return 4
    elif action == (2, 0):  # Move down and left
        return 5
    elif action == (1, 2):  # Move right
        return 6
    elif action == (2, 1):  # Move down
        return 7
    elif action == (2, 2):  # Move down and right
        return 8
    else:
        raise ValueError("Action not in action space")

def action_to_hot_gpu(action, device):
    if torch.eq(action, torch.tensor([[0], [0]], dtype=torch.int, device=device)).all():  # Move left and up
        return torch.tensor((0), dtype=torch.int64, device=device)
    elif torch.eq(action, torch.tensor([[0], [1]], dtype=torch.int, device=device)).all():  # Move up
        return torch.tensor((1), dtype=torch.int64, device=device)
    elif torch.eq(action, torch.tensor([[1], [0]], dtype=torch.int, device=device)).all():  # Move left
        return torch.tensor((2), dtype=torch.int64, device=device)
    elif torch.eq(action, torch.tensor([[1], [1]], dtype=torch.int, device=device)).all():  # stand still
        return torch.tensor((3), dtype=torch.int64, device=device)
    elif torch.eq(action, torch.tensor([[0], [2]], dtype=torch.int, device=device)).all():  # Move right and up
        return torch.tensor((4), dtype=torch.int64, device=device)
    elif torch.eq(action, torch.tensor([[2], [0]], dtype=torch.int, device=device)).all():  # Move left and down
        return torch.tensor((5), dtype=torch.int64, device=device)
    elif torch.eq(action, torch.tensor([[1], [2]], dtype=torch.int, device=device)).all():  # Move right
        return torch.tensor((6), dtype=torch.int64, device=device)
    elif torch.eq(action, torch.tensor([[2], [1]], dtype=torch.int, device=device)).all():  # Move down
        return torch.tensor((7), dtype=torch.int64, device=device)
    elif torch.eq(action, torch.tensor([[2], [2]], dtype=torch.int, device=device)).all():  # Move right and down
        return torch.tensor((8), dtype=torch.int64, device=device)
    else:
        raise ValueError("Action not in action space")

def hot_to_action(hot):
    # Pease reverse the logic presented in action_to_hot:
    if hot == 0:
        return (0, 0)
    elif hot == 1:
        return (0, 1)
    elif hot == 2:
        return (1, 0)
    elif hot == 3:
        return (1, 1)
    elif hot == 4:
        return (0, 2)
    elif hot == 5:
        return (2, 0)
    elif hot == 6:
        return (1, 2)
    elif hot == 7:
        return (2, 1)
    elif hot == 8:
        return (2, 2)
    else:
        raise ValueError("Action not in action space")


def hot_to_action_gpu(hot, device):
    # Pease reverse the logic presented in action_to_hot:
    if hot == torch.tensor((0), dtype=torch.int, device=device):
        return torch.tensor([[0], [0]], dtype=torch.int, device=device)
    elif hot == torch.tensor((1), dtype=torch.int, device=device):
        return torch.tensor([[0], [1]], dtype=torch.int, device=device)
    elif hot == torch.tensor((2), dtype=torch.int, device=device):
        return torch.tensor([[1], [0]], dtype=torch.int, device=device)
    elif hot == torch.tensor((3), dtype=torch.int, device=device):
        return torch.tensor([[1], [1]], dtype=torch.int, device=device)
    elif hot == torch.tensor((4), dtype=torch.int, device=device):
        return torch.tensor([[0], [2]], dtype=torch.int, device=device)
    elif hot == torch.tensor((5), dtype=torch.int, device=device):
        return torch.tensor([[2], [0]], dtype=torch.int, device=device)
    elif hot == torch.tensor((6), dtype=torch.int, device=device):
        return torch.tensor([[1], [2]], dtype=torch.int, device=device)
    elif hot == torch.tensor((7), dtype=torch.int, device=device):
        return torch.tensor([[2], [1]], dtype=torch.int, device=device)
    elif hot == torch.tensor((8), dtype=torch.int, device=device):
        return torch.tensor([[2], [2]], dtype=torch.int, device=device)
    else:
        raise ValueError("Action not in action space")
