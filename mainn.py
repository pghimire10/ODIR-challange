import pandas as pd 
import os
from sklearn.model_selection import train_test_split

BASE_DIR = r"Data"
label = r"Data\ODIR-5K_Training_Annotations(Updated)_V2.xlsx"
df = pd.read_excel(label)
csv_data = df.to_csv(os.path.join(BASE_DIR, "Data.csv"))

left_eye = df[['Left-Fundus', 'Left-Diagnostic Keywords']].copy()
left_eye.columns = ['Image', 'Labels']


left_eye.to_csv(os.path.join(BASE_DIR, "left_eye.csv"))

right_eye = df[['Right-Fundus', 'Right-Diagnostic Keywords']].copy()
right_eye.columns = ['Image', 'Labels']

right_eye.to_csv(os.path.join(BASE_DIR, "right_eye.csv"))
right_eye_path = r"Data\right_eye.csv"

# dir = os.listdir(right_eye_path)


# print(dir)

# get unique diagnostic keywords
keywords = [ keyword  for keywords in df['Left-Diagnostic Keywords'] for keyword in keywords.split('，')]
unique_keywords = set(keywords)

# print(keywords[:10])
# print(unique_keywords)
# print(len(unique_keywords),len(keywords))

# create a mapping from keywords to class labels
class_labels = ['N','D','G','C','A','H','M','O']
keyword_label_mapping  = {
    'normal':'N',
    'retinopathy':'D',
    'glaucoma':'G',
    'cataract':'C',
    'macular degeneration':'A',
    'hypertensive':'H',
    'myopia':'M',
    'lens dust' : 'O',
    'optic disk photographically invisible':'O', 
    'low image quality':'O', 
    'image offset':'O',
}
non_decisive_labels = ["lens dust", "optic disk photographically invisible", "low image quality", "image offset"]

# if the keyword contains label outside of the above then, label them as others 'O'
def generate_individual_label(diagnostic_keywords):
    
    keywords = [ keyword  for keyword in diagnostic_keywords.split('，')]
    contains_normal = False
    for k in keywords:
        for label in keyword_label_mapping.keys():
            if label in k:
                if label == 'normal':
                    contains_normal = True # if found a 'normal' keyword, check if there are other keywords but keep in mind that a normal keyword was found
                else:
                    return keyword_label_mapping[label] # found a proper keyword label, use the first occurence

    # did not find a proper keyword label, see if there are labels other than non-decisive labels, if so, categorize them as 'others'
    decisive_label = False
    for k in keywords:
        if k not in non_decisive_labels and (('normal' not in k) or ('abnormal' in k)):
            decisive_label = True
    if decisive_label:
        # contains decisive label other than the normal and abnormal categories
        return 'O' 
    if contains_normal:
        return 'N'
    
    
    # if any of the above criteria do not match, then return as is
    return keywords[0] # useful for diagnostics, check if there are cases that are not covered by the above

# generate_individual_label('normal fundus'),generate_individual_label('lens dust，drusen，normal fundus	')
df['Left-label']= df['Left-Diagnostic Keywords'].apply(generate_individual_label)
df['Right-label'] = df['Right-Diagnostic Keywords'].apply(generate_individual_label)
df[df['Left-label'].isin(non_decisive_labels)]



left_data = pd.read_csv(r"Data\left_eye.csv")
# print(left_data)
left_column = 'left_labels'
l=[]

for left in left_data['Labels']:
    out = generate_individual_label(left)
    l.append(out)
print(l)

left_data['left_columns']=l
left_data.to_csv(r"Data\left_eye.csv", index="False")


# write test cases
# if both left and right are normal, then the final diagnosis is also normal
def test_normal(row):
    l,r = row['Left-label'], row['Right-label']
    if l == 'N' and r == 'N' and row['N'] != 1:
        return False
    else:
        return True

def test_others(row):
    l,r = row['Left-label'], row['Right-label']
    if row['O'] == 1:
        if l == 'O' or r == 'O':
            return True
        else:
            return False 
    return True

# find rows where both left and right have beeen processed as Normal, but the final diagnosis is not 'N
df[df.apply(test_normal, axis=1) == False]
# find rows where none of the left and right have been processed as Others, but the final diagnosis also contains 'O'
df[df.apply(test_others,axis=1) == False]

manually_remove = ['2174_right.jpg',
'2175_left.jpg',
'2176_left.jpg',
'2177_left.jpg',
'2177_right.jpg',
'2178_right.jpg',
'2179_left.jpg',
'2179_right.jpg',
'2180_left.jpg',
'2180_right.jpg',
'2181_left.jpg',
'2181_right.jpg',
'2182_left.jpg',
'2182_right.jpg',
'2957_left.jpg',
'2957_right.jpg',
]

df = df[(~df['Left-Fundus'].isin(manually_remove)) & (~df['Right-Fundus'].isin(manually_remove))]

# create a new dataframe where each row corresponds to one image
left_fundus = df['Left-Fundus']
left_label = df['Left-label']
left_keywords = df['Left-Diagnostic Keywords']
right_fundus = df['Right-Fundus']
right_label = df['Right-label']
right_keywords = df['Right-Diagnostic Keywords']
id = df['ID']
age = df['Patient Age']
sex = df['Patient Sex']

# separate train and test split

SEED = 234
id_train, id_val = train_test_split(id,test_size=0.1,random_state=SEED)

train_left_fundus = df[df['ID'].isin(id_train)]['Left-Fundus']
train_left_label = df[df['ID'].isin(id_train)]['Left-label']
train_left_keywords = df[df['ID'].isin(id_train)]['Left-Diagnostic Keywords']

train_right_fundus = df[df['ID'].isin(id_train)]['Right-Fundus']
train_right_label = df[df['ID'].isin(id_train)]['Right-label']
train_right_keywords = df[df['ID'].isin(id_train)]['Right-Diagnostic Keywords']


val_left_fundus = df[df['ID'].isin(id_val)]['Left-Fundus']
val_left_label = df[df['ID'].isin(id_val)]['Left-label']
val_left_keywords = df[df['ID'].isin(id_val)]['Left-Diagnostic Keywords']

val_right_fundus = df[df['ID'].isin(id_val)]['Right-Fundus']
val_right_label = df[df['ID'].isin(id_val)]['Right-label']
val_right_keywords = df[df['ID'].isin(id_val)]['Right-Diagnostic Keywords']

# stack left and right columns vertically
train_fundus = pd.concat([train_left_fundus, train_right_fundus],axis=0,ignore_index=True,sort=True)
train_label = pd.concat([train_left_label,  train_right_label],axis=0,ignore_index=True,sort=True)
train_keywords = pd.concat([train_left_keywords,train_right_keywords],axis=0,ignore_index=True,sort=True)

val_fundus = pd.concat([val_left_fundus, val_right_fundus],axis=0,ignore_index=True)
val_label = pd.concat([val_left_label,val_right_label],axis=0,ignore_index=True)
val_keywords = pd.concat([val_left_keywords,val_right_keywords],axis=0,ignore_index=True)

train_df_left_right_separate_row = pd.concat([train_fundus,
                                              train_label,
                                              train_keywords],axis=1,sort=True,
                                              keys = ['fundus','label','keywords']) # stack horizontally
val_df_left_right_separate_row = pd.concat([  val_fundus,
                                              val_label,
                                              val_keywords],axis=1,sort=True,
                                              keys=['fundus','label','keywords']) # stack horizontally


cleaned_train_df = train_df_left_right_separate_row.drop(train_df_left_right_separate_row[train_df_left_right_separate_row['label'].isin(non_decisive_labels)].index)


cleaned_val_df = val_df_left_right_separate_row.drop(val_df_left_right_separate_row[val_df_left_right_separate_row['label'].isin(non_decisive_labels)].index)

cleaned_train_df.to_csv(r'Data\train.csv')
cleaned_val_df.to_csv(r'Data\validation.csv')












