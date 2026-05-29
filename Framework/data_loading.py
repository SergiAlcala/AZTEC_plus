
def get_config(city,ETA, LOOKBACK, DELAY_Block1_Block2,DELAY_Helper, PHI, GAMMA, NUM_SERV_B1,NUM_SERV_H,SEL_SERV_H,
 EPOCHS_block1,EPOCHS_block2,EPOCHS_Helper, B,ALPHA,Simulations,ppf_helper=0.99,ppf_static=0.9):
    config={
        'city': city,
        'ETA': ETA,        
        'LOOKBACK': LOOKBACK,
        'DELAY_Block1_Block2': DELAY_Block1_Block2,
        'DELAY_Helper': DELAY_Helper,
        'PHI': PHI,
        'GAMMA': GAMMA,
        'NUM_SERV_Block_1': NUM_SERV_B1,
        'NUM_SERV_Helper': NUM_SERV_H,
        'SEL_SERV_H':SEL_SERV_H,
        'EPOCHS_block1': EPOCHS_block1,
        'EPOCHS_block2': EPOCHS_block2,
        'EPOCHS_Helper': EPOCHS_Helper,
        'B': B,
        'ALPHA': ALPHA,
        'Simulations': Simulations,
        'ppf_helper': ppf_helper,
        'ppf_static': ppf_static
    }
    return config  