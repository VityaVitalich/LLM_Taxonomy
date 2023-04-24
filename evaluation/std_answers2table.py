import pandas as pd

def answers(str_ans, name_of_pattern, varied_param):
    str_ans += name_of_pattern
    columns_name = []
    values = []
    for ind, metrics in enumerate(str_ans.split('\n')):
        
        if ind == 6:
            _name = varied_param
            number = name_of_pattern
        else:
            _name, number = metrics.split(' ')
            number = round(float(number), 5)
            _name = _name[:-1]
        
        columns_name.append(_name)
        values.append([number])
        
    
        
    df = pd.DataFrame(values).T
    df.columns = columns_name
    df.set_index(varied_param, inplace=True)
    return df