import pandas as pd

def custom_prt_prediction_validator(df_prt):
    # Define keywords for 'late' reviews
    keywords = 'lama|lambat|late|telat|waited|too long|dingin|nyasar'

    # Create a new column 'total_delay' representing the sum of 'pick_late' and 'drop_late'
    df_prt['total_delay'] = df_prt['pick_late'].add(df_prt['drop_late'], fill_value=0)

    # LOGIC 1: 
    # If prediction either 'not_responding_to_their_job_accordingly' and 'not_confirming_the_delivery_address', 
    # and either pick or drop late >= 10, change its confidence score to 1, 
    # and #QC remark only if the prediction is 'not_responding_to_their_job_accordingly' and review is short
    mask_1 = ((df_prt['prediction'] == 'not_responding_to_their_job_accordingly') |
              (df_prt['prediction'] == 'not_confirming_the_delivery_address')) & \
              (df_prt['total_delay'].notna() & (df_prt['total_delay'] >= 10))

    df_prt.loc[mask_1, 'conf_score'] = 1
    mask_1_qc = mask_1 & (df_prt['prediction'] == 'not_responding_to_their_job_accordingly') & (df_prt['review_or_remarks'].str.len() < 30)
    df_prt.loc[mask_1_qc, 'review_or_remarks'] += " #QC: " + df_prt.loc[mask_1_qc, 'total_delay'].astype(str).str.rstrip('.0') + " mins late"

    # LOGIC 2:
    #   IF prediction is 'not_responding_to_their_job_accordingly' and sum of pick_late and drop_late is less than 0,
    #   change prediction to 'Unclear' and add remarks #QC: Dax {pick_late+drop_late} mins early
    mask_2 = (df_prt['prediction'] == 'not_responding_to_their_job_accordingly') & (df_prt['total_delay'] < 0)
    
    df_prt.loc[mask_2, 'prediction'] = 'unclear'
    df_prt.loc[mask_2, 'review_or_remarks'] += " #QC: Dax " + \
        df_prt.loc[mask_2, 'total_delay'].astype(str).apply(lambda x: x.lstrip('-').rstrip('.0')) + " mins early"

    df_prt.loc[mask_2, 'conf_score'] = 1

    # LOGIC 3:
    # If prediction is 'product_related' and sum of pick_late and drop_late is greater than or equal to 30,
    # change prediction to 'not_responding_to_their_job_accordingly' and add mins late remarks
    mask_3 = (df_prt['prediction'] == 'product_related') & (df_prt['total_delay'] >= 30)
    
    df_prt.loc[mask_3, 'prediction'] = 'not_responding_to_their_job_accordingly'
    df_prt.loc[mask_3, 'review_or_remarks'] += " #QC: " + \
        df_prt.loc[mask_3, 'total_delay'].astype(str).str.rstrip('.0') + " mins late"

    df_prt.loc[mask_3, 'conf_score'] = 1

    # LOGIC 4:
    # If prediction is 'unclear' or 'mex_related' and sum of pick_late and drop_late is greater than or equal to 10,
    # and review contains keywords, change prediction to 'not_responding_to_their_job_accordingly' and add mins late remarks
    mask_4 = ((df_prt['prediction'].isin(['unclear', 'mex_related'])) & 
              (df_prt['total_delay'] >= 10) &
              (df_prt['review_or_remarks'].str.contains(keywords, case=False, na=False, regex=True)))

    df_prt.loc[mask_4, 'prediction'] = 'not_responding_to_their_job_accordingly'
    df_prt.loc[mask_4, 'review_or_remarks'] += " #QC: " + \
        df_prt.loc[mask_4, 'total_delay'].astype(str).str.rstrip('.0') + " mins late"

    df_prt.loc[mask_4, 'conf_score'] = 1

    # LOGIC 5:
    # If 'review_or_remarks' contains keyword and 'is_batching' is 1 and 'total_delay' is less than or equal to 10,
    # change prediction to 'Product Related' and add remarks '#QC: Double Order'
    mask_5 = (df_prt['review_or_remarks'].str.contains(keywords, case=False, na=False, regex=True)) & \
             (df_prt['is_batching'] == 1) & \
             (df_prt['total_delay'] <= 10)

    df_prt.loc[mask_5, 'prediction'] = 'Product Related'
    df_prt.loc[mask_5, 'review_or_remarks'] += " #QC: Double Order " + \
        df_prt.loc[mask_5, 'total_delay'].astype(str).str.rstrip('.0') + " mins late"

    df_prt.loc[mask_5, 'conf_score'] = 1

    return df_prt
