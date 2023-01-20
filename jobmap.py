import pandas as pd
import numpy as np
import random
import json
from scipy.optimize import linear_sum_assignment
import requests, zipfile, io

def read_excel_zip_url(url, filepath):
    r = requests.get(url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    file=z.read(filepath)
    return pd.read_excel(io.BytesIO(file))

def get_soc_code_map(SOC_CODES_LOC='https://www.bls.gov/soc/2018/soc_2018_definitions.xlsx'):
    soc_codes=pd.read_excel(SOC_CODES_LOC, header=7)
    soc_codes['SOC Code']=soc_codes.apply(lambda row: row['SOC Code'].replace('-', ''), axis=1)
    soc_code_map={row['SOC Code']: row['SOC Title'] for ind, row in soc_codes.iterrows()}
    return soc_code_map

def stochastic_round(vec):
    whole_part=vec.astype(int)
    dec_part=vec-whole_part
    return whole_part+(dec_part>np.random.uniform(0, 1, len(vec)))

def get_indices_master_list(soc, master_list):
    # start with full 6 digits- if its a detailed code already, just return this in a list
    # otherwise remove digits as necessary until 
    # we find a list of detailed codes which start with this stem
    part_soc=soc
    soc_indices=[0]
    while sum(soc_indices)==0:
        soc_indices=[soc.startswith(part_soc) for soc in master_list]
        part_soc=part_soc[:-1]
    return soc_indices 


class JobMap():
    def __init__(self, state_code, bls_data=None):
        if bls_data is None:
            bls_data=read_excel_zip_url(
                url='https://www.bls.gov/oes/special-requests/oesm21all.zip',
                filepath='oesm21all/all_data_M_2021.xlsx')
        self.create_soc_to_naics_matrix(bls_data)
        self.get_state_soc_salary_map(state_code, bls_data)
    
    def get_state_soc_salary_map(self, state_code, bls_data):
        # Maybe need to have 1 more mapping for major or minor codes if fails on any detailed codes
        # First create the mapping from SOC to average salary for all states combined (in case some are missing for the state)
        all_states_emp_data=bls_data.loc[bls_data['AREA_TYPE']==2]
        all_states_emp_data=all_states_emp_data.loc[pd.to_numeric(all_states_emp_data['A_MEDIAN'], errors='coerce').notnull()]
        all_states_emp_data=all_states_emp_data.loc[pd.to_numeric(all_states_emp_data['TOT_EMP'], errors='coerce').notnull()]
        all_states_emp_data['OCC_CODE']=all_states_emp_data['OCC_CODE'].apply(lambda x: x.replace('-', ''))
        all_states_emp_data['PROD_EMP_MEDIAN']=all_states_emp_data['A_MEDIAN']*all_states_emp_data['TOT_EMP']
        all_states_salary_by_soc=all_states_emp_data.groupby('OCC_CODE').agg(
            {'PROD_EMP_MEDIAN': 'sum', 'TOT_EMP': 'sum'})            
        all_states_salary_by_soc['WEIGHTED_MEDIAN']=all_states_salary_by_soc['PROD_EMP_MEDIAN']/all_states_salary_by_soc['TOT_EMP']
        all_states_soc_to_salary={ind: row['WEIGHTED_MEDIAN'] for ind, row in all_states_salary_by_soc.iterrows()}

#         self.soc_to_salary_state={row['OCC_CODE']: row['A_MEDIAN'] for ind, row in area_emp_data.iterrows()}
        # Then get the mapping for this speific state and combine
        this_state_emp_data=all_states_emp_data.loc[all_states_emp_data['PRIM_STATE']==state_code]
        this_state_salary_by_soc=this_state_emp_data.groupby('OCC_CODE').agg(
            {'PROD_EMP_MEDIAN': 'sum', 'TOT_EMP': 'sum'})            
        this_state_salary_by_soc['WEIGHTED_MEDIAN']=this_state_salary_by_soc['PROD_EMP_MEDIAN']/this_state_salary_by_soc['TOT_EMP']
        this_state_soc_to_salary={ind: row['WEIGHTED_MEDIAN'] for ind, row in this_state_salary_by_soc.iterrows()}
        
        combined_soc_to_salary=all_states_soc_to_salary
        for soc in this_state_soc_to_salary:
            combined_soc_to_salary[soc]=this_state_soc_to_salary[soc]       
        self.soc_to_salary_state=combined_soc_to_salary       
    
    def create_soc_to_naics_matrix(self, bls_data, soc_level='detailed'):
        bls_ind=bls_data.loc[bls_data['I_GROUP'].isin(['3-digit', '4-digit', '5-digit', '6-digit'])]
        bls_ind=bls_ind.loc[((bls_ind['O_GROUP']==soc_level)
                                   &(bls_ind['TOT_EMP']!='**'))].copy()
        bls_ind = bls_ind.astype({'TOT_EMP': 'float', 'NAICS': str})
        bls_ind['OCC_CODE']=bls_ind['OCC_CODE'].apply(lambda x: x.replace('-', ''))
        emp_by_soc_naics=bls_ind.groupby(['OCC_CODE','NAICS']
                                           )['TOT_EMP'].mean().reset_index()
        soc_naics_pivot=pd.pivot(data=emp_by_soc_naics, index='OCC_CODE', columns='NAICS', values='TOT_EMP').fillna(0)
#         soc_naics_pivot.columns=[str(c) for c in soc_naics_pivot.columns]
        soc_to_naics_df_norm=soc_naics_pivot.div(soc_naics_pivot.sum(axis=0), axis=1)
        self.soc_to_naics_df_norm=soc_to_naics_df_norm
        self.soc_order=list(soc_to_naics_df_norm.index)
        self.naics_order=list(soc_to_naics_df_norm.columns)
        self.soc_to_naics_mat=np.array(soc_to_naics_df_norm)
        
    def get_employees_by_soc(self, naics_dict, as_dict=True):
        naics_employees=np.zeros(self.soc_to_naics_mat.shape[1])
        for naics in naics_dict:
            # Try to match the full naics code- if none found, discard digits one by one
            # use minimum of 2 digits
            ind_list=[ind for ind in range(len(self.naics_order)) if self.naics_order[ind].startswith(naics)]
            digits_ignored=1
            while ((len(ind_list)==0) and (len(naics)-digits_ignored>=2)):
                naics_stem=(naics[:-digits_ignored])
                ind_list=[ind for ind in range(len(self.naics_order)) if self.naics_order[ind].startswith(naics_stem)]
                digits_ignored+=1
            if len(ind_list)>0:
                naics_employees[ind_list]=naics_dict[naics]/len(ind_list)
            else:
                print('Error: naics {}'.format(naics))
                
        soc_employees=np.dot(self.soc_to_naics_mat, naics_employees)
        soc_employees_int=stochastic_round(soc_employees)
        if as_dict:
            output={}
            for i in range(len(soc_employees_int)):
                if soc_employees_int[i]>0:
                    output[self.soc_order[i]]=soc_employees_int[i]
            return output
        return soc_employees_int
    
    def get_job_list(self, soc_dict):
        soc_list=[]
        salary_list=[]
        for soc in soc_dict:
            soc_list.extend([soc]*soc_dict[soc])
            try:
                salary=self.soc_to_salary_state[soc]
            except:
                print('{}: No salary info'.format(soc))
                salary=None
            salary_list.extend([salary]*soc_dict[soc])
        return {'soc': soc_list, 'salary': salary_list} 

class JobTransitioner():
    def __init__(self):
        self.get_change_matrix()

    def get_change_matrix(self):
        print('Downloading career changers matrix from https://www.onetcenter.org')
        change_df=pd.read_excel('https://www.onetcenter.org/dl_files/database/db_20_3_excel/Career%20Changers%20Matrix.xlsx')
        change_df['O*NET-SOC Code']=change_df['O*NET-SOC Code'].apply(lambda x: x.replace('-', '').split('.')[0])
        change_df['Related O*NET-SOC Code']=change_df['Related O*NET-SOC Code'].apply(lambda x: x.replace('-', '').split('.')[0])
        change_matrix = pd.crosstab(change_df['O*NET-SOC Code'], change_df['Related O*NET-SOC Code'])
        self.master_list=change_matrix.index
        missing_to_cols=[ind for ind in self.master_list if ind not in change_matrix.columns]
        for col in  missing_to_cols:
            change_matrix.loc[:,col]=0
        for ind in change_matrix.index: # every job can transition to the same job
            change_matrix.loc[ind, ind]=1
        self.change_matrix=change_matrix[self.master_list]>0
        
    def soc_list_to_matrix(self, soc_list):
        binary_mat=np.array([get_indices_master_list(soc, self.master_list) for soc in soc_list])
        int_mat=binary_mat.astype(int)
        norm_mat=int_mat/int_mat.sum(axis=1)[:, np.newaxis]
        return norm_mat
    
    def get_salary_decrease_mat(self, from_salaries, to_salaries):
        salary_diff= np.array(from_salaries).reshape(
            [len(from_salaries),1])-np.array(to_salaries).reshape(
            [1,len(to_salaries)])
        salary_decrease=salary_diff>0
        return salary_decrease
    
    def get_cost_matrix(self, from_soc_mat,  to_soc_mat, salary_decrease):
        person_job_trans_mat =  np.dot(
            np.dot(from_soc_mat, self.change_matrix),
            to_soc_mat.T)
        cost_mat=1e10*(person_job_trans_mat<0.5).astype(int) + 1e10*(salary_decrease).astype(int)
        return cost_mat
    
    def match_jobs(self, from_soc_mat,  to_soc_mat, salary_decrease, n=None):
        cost_matrix=self.get_cost_matrix(from_soc_mat,  to_soc_mat, salary_decrease)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        transitions=pd.DataFrame({'row_ind': row_ind, 'col_ind': col_ind, 
                     'cost': [cost_matrix[row_ind[i], col_ind[i]] for i in range(len(row_ind))]})
        valid_transitions=transitions.loc[transitions['cost']<1e9].copy()
        if n is not None:
            valid_transitions=valid_transitions.sort_values('cost', ascending=True).iloc[:n]
        return valid_transitions

