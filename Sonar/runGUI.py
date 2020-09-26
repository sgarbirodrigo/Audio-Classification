import os

from Sonar.AI import AI

ai = AI(src_root="/Volumes/My Passport/HD externo/1 - Projetos/6 - Sonar/cleaned_dataset/", sample_rate=16000, delta_time=1,
        batch_size=16, n_fdt=512)

ai.train()


