import torch
from binary_thinning_3d import binary_thinning


def test_thinning():
    # Create a 5x5x5 block and put a 3x3x3 cube in the middle
    t = torch.zeros((5, 5, 5), dtype=torch.uint8, device="cuda")
    t[1:4, 1:4, 1:4] = 1

    print("Original sum:", t.sum().item())

    # Run thinning
    binary_thinning(t)

    print("Thinned sum:", t.sum().item())

    print("Coordinates of remaining points:")
    coords = torch.nonzero(t)
    for c in coords:
        print(c.tolist())

    print("Center pixel value:", t[2, 2, 2].item())
    assert t.sum().item() == 3
    assert t[2, 2, 2].item() == 1
    print("Test passed!")


if __name__ == "__main__":
    test_thinning()
