# RvQuality

## Synopsis
A python library that establishes the quality of a review. Based on Luca D'Amato's thesis.

## Code Example
```python
#!/usr/bin/env python3
from rvquality.quality_table import QualityTable
from rvquality.options import Options
import rvquality.components as components
import pandas as pd

main_table = pd.read_csv('path_to_csv_data.csv', sep=';')
# sample id from main table
rv_id = main_table[qt.opts.ID_NAME][0]

# components to include
components = [
  components.C1(), 
  components.C2(), 
  components.C3(), 
  components.C4(), 
  components.C7(), 
  components.C8(),
  components.C9(),
  components.C10(),
  components.C11(),
  components.C12()]

# sets options
opts = Options()
opts.ID_NAME = "id"

quality_table = QualityTable(data_frame)
quality_table.prepare()
qt.quality_of(rv_id, components)
```

## Installation
```bash
# clone the repository
git clone https://gitlab.com/Eskilop/rvquality rvquality

# cd into project directory
cd rvquality

# install
python3 setup.py install
```