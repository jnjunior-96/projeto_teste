
from scipy.stats import (
    levene,
    mannwhitneyu,
    ttest_ind,
)
from collections import namedtuple

def analise_levene(dataframe, alfa=0.05, center='mean'):
    print('Teste de Levene')

    estatistica, valor_p_levene =levene(*[dataframe[coluna] for coluna in dataframe.columns], center=center ,nan_policy='omit')

    print(f'{estatistica:.3f}')
    if valor_p_levene > alfa:
        print(f'Variancias iguais (valor p : {valor_p_levene:.3f})')
    else:
        print(f'Ao menos uma varianca é diferente (valor p : {valor_p_levene:.3f})')


def analise_ttest_ind(dataframe, alfa=0.05, variancias_iguais=True, alternativa='two-sided'):

    print('Teste t de Student')


    estatistica_ttest, valor_p_ttest = ttest_ind(
        *[dataframe[coluna] for coluna in dataframe.columns],
        equal_var=variancias_iguais,
        alternative=alternativa,
        nan_policy='omit')

    print(f'{estatistica_ttest=:.3f}')
    if valor_p_ttest > alfa:
        print(f'Não rejeita a hipótese nula (valor p : {valor_p_ttest:.3f})')
    else:
        print(f'Rejeita a hipótese nula (valor p : {valor_p_ttest:.3f})')


def analise_mannwhitneyu(dataframe, alfa=0.05, alternativa='two-sided'):
    print('Teste Mannwhitneyu')

    estatistica_mannwhitneyu, valor_p_mannwhitneyu = mannwhitneyu(
        *[dataframe[coluna] for coluna in dataframe.columns],
        nan_policy='omit', alternative=alternativa)

    print(f'{estatistica_mannwhitneyu=:.3f}')
    if valor_p_mannwhitneyu > alfa:
        print(f'Não rejeita a hipótese nula (valor p : {valor_p_mannwhitneyu:.3f})')
    else:
        print(f'Rejeita a hipótese nula (valor p : {valor_p_mannwhitneyu:.3f})')

