from helper import prepare_model

if __name__ == '__main__':
    names = [
        'mmaesr-flowers-good',
        'mmaesr-pets-good',
        'maesr-flowers-good',
        'maesr-pets-good',
        'maesr_1p-flowers',
        'maesr_1p-pets',
        'mmaesr_1p-flowers',
        'mmaesr_1p-pets'
        ]
    models = [prepare_model(f'states/{name}/', 'checkpoint.pth', name.split('-')[0].split('_')[0]) for name in names]
    for name, model in zip(names, models):
        model_total_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        print(f'{name} has {model_total_params:.0f} trainable parameters')
    for name, parameter in zip(model.named_parameters(), model.parameters()):
        if parameter.requires_grad:
            print('These are counted:', name[0])
        else:
            print('These are not counted:', name[0])