# AutoAugment
```bash
git clone https://github.com/R300-AI/AutoAugment.git
pip install -r AutoAugment/requirements.txt
```
```python
from AutoAugment.augmentors import ObjectAugmentor

augmentor = ObjectAugmentor(maximum_size = 16, transform_option = [A.Blur(p=0.8)], ignore_classes = ['battery'], maximum_process_second = 60 * 60 * 1)
NEW_DATASET_PATH = augmentor.fit(DATASET_PATH, save=True, verbose=True)
```
