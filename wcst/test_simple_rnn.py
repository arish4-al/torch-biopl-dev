"""Minimal script to test WCST training with a simple RNN (no SpatiallyEmbeddedRNN)."""
import torch

from hyperparameters import get_default_hp
from task import WCST, get_default_hp_wcst


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hp, _, loss_fnc = get_default_hp()


class SimpleWCSTRNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size=64, n_output=3, n_output_rule=2):
        super().__init__()
        self.rnn = torch.nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=False,
            nonlinearity="tanh",
        )
        self.readout = torch.nn.Linear(hidden_size, n_output + n_output_rule)
        self.n_output = n_output

    def forward(self, x):
        out, _ = self.rnn(x)
        logits = self.readout(out)
        return logits[..., : self.n_output], logits[..., self.n_output :]


def main():
    model = SimpleWCSTRNN(
        input_size=hp["n_input"],
        hidden_size=64,
        n_output=hp["n_output"],
        n_output_rule=hp["n_output_rule"],
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=hp["learning_rate"])
    hp_wcst = get_default_hp_wcst()
    rule_list = ["rule1", "rule2"]

    for batch_idx in range(15):
        rule = rule_list[batch_idx % 2]
        wcst = WCST(
            hp=hp,
            hp_wcst=hp_wcst,
            rule=rule,
            rule_list=rule_list,
            n_features_per_rule=2,
            n_test_cards=3,
        )
        x, _, yhat, yhat_rule, _ = wcst.make_task_batch(batch_size=hp["batch_size"])
        x = x.to(device)
        yhat = yhat.to(device)
        yhat_rule = yhat_rule.to(device)

        y, y_rule = model(x)
        loss = loss_fnc(y, yhat) + loss_fnc(y_rule, yhat_rule)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            resp_correct, _, _ = wcst.get_perf(y.cpu(), yhat.cpu())
            rule_correct, _, _ = wcst.get_perf_rule(y_rule.cpu(), yhat_rule.cpu())
        print(
            f"Batch {batch_idx + 1}: loss={loss.item():.4f} "
            f"choice_acc={resp_correct.float().mean().item():.3f} "
            f"rule_acc={rule_correct.float().mean().item():.3f}"
        )
    print("Training procedure OK.")


if __name__ == "__main__":
    main()
