import pandas as pd 
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer  
from sklearn.preprocessing import OneHotEncoder
from category_encoders import TargetEncoder
import category_encoders as ce




class pre_processing :
    def __init__(self) -> None:
        pass
    
    
    def pre_processing(self, df :pd.DataFrame, train : bool, categorical_var_OHE:list,
                       categorical_var_OrdinalEncoding:dict, categorical_var_TE: list, target, continious_var:list,
                       encoding_type_cont) -> pd.DataFrame :
        """
        Summary: This method aim to encode and scale a dataframe (df)

        Args:
            df (pd.DataFrame): cleaned dataframe
            train (bool): Ask if you are encoding/scaling a traning set or not (if yes we fit_transform it, else we transform it)
            categorical_var_OHE (list): list of categorical columns who will be encoded with OHE
                                        Ex : ["col1", "col2"]
            categorical_var_OrdinalEncoding (dict): dict of categorical columns who will be encoded with ordinal encoding.
                                                    Ex : {'col1': {'a':0, 'b':1, 'c':2},
                                                          'col2': {'d': 2, 'e':1, 'f':0}}
            categorical_var_TE (list): list of categorical columns who will be encoded with Target encoder
                                        Ex : ["col1", "col2"]
            target (series): y_train for training TE
            continious_var (list): list of continious columns who will be scaled with "encoding_type_cont"
                                    Ex : ["col1", "col2"]
            encoding_type_cont (_type_): types of scaling. MinMaxScaler() or StandardScaler()

        Returns:
            pd.DataFrame: encoded/scaled dataframe
        """
        df_pre_processed = df.copy()
        
        if train == True :
            #continious var encoding :
            if len(continious_var) != 0 :
                #continious var :
                self.scaler = encoding_type_cont #StandardScaler() #or MinMaxScaler()
                df_pre_processed[continious_var] = self.scaler.fit_transform(df_pre_processed[continious_var])    
           
            #categorical encoding :
            if len(categorical_var_OHE) != 0 :
            #categorical var : OHE
                self.enc_OHE = OneHotEncoder(drop='first', sparse=False).fit(df_pre_processed[categorical_var_OHE])
                encoded = self.enc_OHE.transform(df_pre_processed[categorical_var_OHE])
                encoded_df = pd.DataFrame(encoded,columns=self.enc_OHE.get_feature_names_out())
                df_pre_processed.drop(categorical_var_OHE, axis=1, inplace=True)
                df_pre_processed = pd.concat([df_pre_processed, encoded_df], axis=1)
               
            if len(categorical_var_OrdinalEncoding) != 0 :
            #categorical var : Ordinal input example -> {"var" : {'c':0,'b':1,'a':2}}
                for i in range(len(categorical_var_OrdinalEncoding)) :
                    var = list(categorical_var_OrdinalEncoding.keys())[i]
                    self.enc_ordinal = ce.OrdinalEncoder(cols=[var], return_df=True, mapping=[{'col':var,'mapping':categorical_var_OrdinalEncoding[var]}])
                    df_pre_processed[var] = self.enc_ordinal.fit_transform(df_pre_processed[var])
            
            if len(categorical_var_TE) != 0 :
            #categorical var : Target encoding
                self.enc_TE = TargetEncoder().fit(df_pre_processed[categorical_var_TE], target)
                encoded_TE = self.enc_TE.transform(df_pre_processed[categorical_var_TE])
                encoded_df_TE = pd.DataFrame(encoded_TE,columns=self.enc_TE.get_feature_names_out())
                df_pre_processed.drop(categorical_var_TE, axis=1, inplace=True)
                df_pre_processed = pd.concat([df_pre_processed, encoded_df_TE], axis=1)
            
       
        else :
            #continious encoding :
            if len(continious_var) != 0 :
                #continious var :
                df_pre_processed[continious_var] = self.scaler.transform(df_pre_processed[continious_var])
             
            #categorical encoding :  
            if len(categorical_var_OHE) != 0 :
                #categorical var : OHE
                encoded2 = self.enc_OHE.transform(df_pre_processed[categorical_var_OHE])
                encoded_df2 = pd.DataFrame(encoded2,columns=self.enc_OHE.get_feature_names_out())
                df_pre_processed.drop(categorical_var_OHE, axis=1, inplace=True)
                df_pre_processed = pd.concat([df_pre_processed, encoded_df2], axis=1)
           
            if len(categorical_var_OrdinalEncoding) != 0 :
            #categorical var : Ordinal input example -> {"var" : {'c':0,'b':1,'a':2}}
                for i in range(len(categorical_var_OrdinalEncoding)) :
                    var = list(categorical_var_OrdinalEncoding.keys())[i]    
                    df_pre_processed[var] = self.enc_ordinal.transform(df_pre_processed[var])
             
            if len(categorical_var_TE) != 0 :
                #categorical var : Target Encoding
                encoded2_TE = self.enc_TE.transform(df_pre_processed[categorical_var_TE])
                encoded_df2_TE = pd.DataFrame(encoded2_TE,columns=self.enc_TE.get_feature_names_out())
                df_pre_processed.drop(categorical_var_TE, axis=1, inplace=True)
                df_pre_processed = pd.concat([df_pre_processed, encoded_df2_TE], axis=1)
            
        return df_pre_processed