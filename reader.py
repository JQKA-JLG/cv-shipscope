from dataset import ShipScopeDataset
from torch.utils.data import DataLoader

if __name__ == "__main__":
    train_dataset = ShipScopeDataset("data/partitioned/train")
    test_dataset = ShipScopeDataset("data/partitioned/test")
    val_dataset = ShipScopeDataset("data/partitioned/validate")

    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    print(f"Train-Dataset expected total: {len(train_dataset)}")
    label_neg_counter = 0
    label_pos_counter = 0
    total = 0
    for features, labels in train_dataloader:
        total += features.shape[0]
        label_neg_counter += (labels == 0).sum().item()
        label_pos_counter += (labels == 1).sum().item()

    print(f"Train-Dataset actual total: {total}")
    print(f"Train-Dataset Label 0: {label_neg_counter}")
    print(f"Train-Dataset Label 1: {label_pos_counter}\n")

    print(f"Test-Dataset expected total: {len(test_dataset)}")
    label_neg_counter = 0
    label_pos_counter = 0
    total = 0
    for features, labels in test_dataloader:
        total += features.shape[0]
        label_neg_counter += (labels == 0).sum().item()
        label_pos_counter += (labels == 1).sum().item()

    print(f"Test-Dataset actual total: {total}")
    print(f"Test-Dataset Label 0: {label_neg_counter}")
    print(f"Test-Dataset Label 1: {label_pos_counter}\n")

    print(f"Val-Dataset expected total: {len(val_dataset)}")
    label_neg_counter = 0
    label_pos_counter = 0
    total = 0
    for features, labels in val_dataloader:
        total += features.shape[0]
        label_neg_counter += (labels == 0).sum().item()
        label_pos_counter += (labels == 1).sum().item()

    print(f"Val-Dataset actual total: {total}")
    print(f"Val-Dataset Label 0: {label_neg_counter}")
    print(f"Val-Dataset Label 1: {label_pos_counter}")