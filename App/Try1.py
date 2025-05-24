# app.py

from shiny import App, ui, render
import pandas as pd
import matplotlib.pyplot as plt
import os

os.getcwd()
os.chdir(r"c:\\Users\\migue\\Documents\\KU_LEUVEN\\7_Semester\\Modern Data Analitycs\\Assigment\\Project\\Modern-Data-Analytics")

# ─── 1) Cargar tu archivo Excel al iniciar la app ─────────
# Ajusta la ruta al archivo dentro de tu proyecto
DATA_PATH = "./features/df_raw.xlsx"
df = pd.read_excel(DATA_PATH)

# 2) Interfaz de usuario
app_ui = ui.page_fluid(
    ui.h2("Gráfico de Barras desde Archivo Pre-cargado"),
    ui.layout_sidebar(
        # ─── Panel lateral ─────────────────────────────
        ui.sidebar(
            ui.input_select(
                "col_cat", 
                "Columna Categoría:",
                choices=list(df.columns),
                selected=df.columns[0]
            ),
            ui.input_select(
                "col_val", 
                "Columna de Valores:",
                choices=list(df.columns),
                selected=df.columns[1]
            ),
            open="open"
        ),
        # ─── Panel principal ────────────────────────────
        ui.output_plot("barplot"),
    )
)

# 3) Lógica del servidor
def server(input, output, session):
    @output
    @render.plot
    def barplot():
        cat = input.col_cat()
        val = input.col_val()
        agg = df.groupby(cat)[val].sum()

        fig, ax = plt.subplots()
        agg.plot.bar(ax=ax)
        ax.set_xlabel(cat)
        ax.set_ylabel(val)
        ax.set_title(f"{val} por {cat}")
        fig.tight_layout()
        return fig

# 4) Crear y ejecutar la app
app = App(app_ui, server)

if __name__ == "__main__":
    app.run()