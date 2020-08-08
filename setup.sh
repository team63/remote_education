# export LC_ALL="es_ES.UTF-8"
# export LC_CTYPE="es_ES.UTF-8"
# sudo dpkg-reconfigure locales

mkdir -p ~/.streamlit/

echo "\
[server]\n\
port = $PORT\n\
enableCORS = false\n\
headless = true\n\
\n\
" > ~/.streamlit/config.toml