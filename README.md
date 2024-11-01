# AutoAugment
```bash
git clone https://github.com/R300-AI/AutoAugment.git
pip install -r AutoAugment/requirements.txt
```
```python
from AutoAugment.augmentors import ObjectAugmentor

augmentor = ObjectAugmentor(maximum_size = 16, maximum_process_second = 60 * 60 * 1)
NEW_DATASET_PATH = augmentor.fit(DATASET_PATH, save=True, verbose=True)
```
