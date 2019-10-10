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
git clone https://github.com/lucadmt/RVQuality.git rvquality

# cd into project directory
cd rvquality

# install
python3 setup.py install
```

## License
```
Copyright (C) 2022 Luca D'Amato

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
```
