sudo cp config/gguf-api.service.example /etc/systemd/system/gguf-api.service
sudo systemctl daemon-reload
sudo systemctl enable gguf-api
sudo systemctl start gguf-api
