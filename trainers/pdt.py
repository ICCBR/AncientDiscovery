from tqdm import tqdm

from trainers.base_trainer import BaseTrainer
from utils import loss_helper, model_helper


class PairedDecoderTrainer(BaseTrainer):
    def _train_epoch(self):
        target_recon_loss, source_recon_loss, dis_loss_all, count = 0.0, 0.0, 0.0, 0.0
        target_data, source_data, labels = self.dataset.random_data(self.target_data, self.source_data, self.labels)
        data = zip(target_data, source_data, labels)
        for target_batch, source_batch, label_batch in tqdm(data, total=len(labels), desc="Training"):
            target_batch, source_batch = target_batch.to(self.device), source_batch.to(self.device)
            target_code, target_loss = model_helper.run_batch(self.model, target_batch, self.core, input_source=False)
            source_code, source_loss = model_helper.run_batch(self.model, source_batch, self.core, input_source=True)
            dis_loss = loss_helper.cal_dis_err(target_code, source_code, label_batch, criterion=self.criterion)
            dis_loss = (target_loss / dis_loss + source_loss / dis_loss) / 2 * dis_loss
            total_loss = target_loss + source_loss + dis_loss
            self._backward(self.optimizer, total_loss)

            # calculate loss here
            count += len(label_batch)
            target_recon_loss += target_loss.item()
            source_recon_loss += source_loss.item()
            dis_loss_all += dis_loss.item()
        return {"Target reconstruct loss": target_recon_loss / count,
                "Source reconstruct loss": source_recon_loss / count, "Distance loss": dis_loss_all / count}
