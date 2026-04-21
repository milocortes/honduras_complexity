# Complejidad Honduras

## Descarga del repositorio

Para descargar el repositorio para la rama ```main``` utiliza la instrucción:

```
git clone https://github.com/milocortes/honduras_complexity.git
```

## Sincronizamos el ambiente virtual de uv

```bash 
uv sync
```

## Ejecución de notebooks de Marimo

Para editar un notebook de Marimo, ejecuta la instrucción:

```bash 
uv run marimo edit complejidad_municipios.py
```

## Resumen de programas

### Recodificación CIIU Rev 4 y NAICS 2022
* Recodificación CIIU Rev 4 a NAICS 2022 `ciiu_rev_4-naics_2022.py`.

### Actividades Transables
* Cálculo de la razon de empleo y establecimiento transables por actividad CIIU Rev 4 recodificada para USA : `actividades_transables_usa.py`. 

### Medidas de Complejidad Honduras (País-Departamento) y USA (MSA)
* Construye medidas de complejidad para Honduras (País y Departamentos) y Zonas Micro y Metropolitanas de USA :  `construye_datos_complejidad_usa_hnd.py`. 
* Construye portafolios de industrias para Honduras : `construye_portafolios_usa_hnd.py`.
* Construye visualizaciones ECI vs {GPD,POB} :  `construye_visualizaciones_complejidad_usa_hnd.py`
* Programa que recodifica actividades : `recodifica_homologa_datos.py`

### Medidas de Complejidad con datos de INDSTAT Revisión 4
* Construye medidas de complejidad usando datos de INDSTAT Revisión 4 : `indstat.py`.

### Medidas de Complejidad con datos de OCDE Structural business statistics by size class and economic activity (ISIC Rev. 4)
* Construye medidas de complejidad usando datos de OCDE Structural business statistics by size class and economic activity (ISIC Rev. 4) `oecd_data.py`.

### Medidas de Complejidad Honduras (País-Departamento), Ecuador, USA (MSA)
* Construye medidas de complejidad para Honduras (País), Ecuador (País) y Zonas Micro y Metropolitanas de USA :  `construye_datos_complejidad_usa_hnd_ecu.py`. 



