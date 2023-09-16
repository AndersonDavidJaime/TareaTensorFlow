package com.example.aplicaciontensorflow;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.os.Build;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;
import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.List;

public class MainActivity extends AppCompatActivity {
    public static int REQUEST_GALLERY = 222;
    public static int REQUEST_CAMERA = 111;
    private TextView txtParecido;
    private TextView txtResults;
    private List<String> classNames;
    private Interpreter interpreter;
    private final int imageSize = 224;
    Button btngaleria;
    ImageView mImageView;
    private Bitmap mSelectedImage;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        txtParecido = findViewById(R.id.txtParecido);
        txtResults = findViewById(R.id.txtresults);
        btngaleria = findViewById(R.id.btGallery);
        mImageView = findViewById(R.id.image_view);
        add_events();
        IniciarModelo();
    }
    private void IniciarModelo() {
        try {
            interpreter = new Interpreter(CargarModelo(this, "modelo3.tflite"));
            classNames = CargarLabel(this, "labels.txt");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
    private MappedByteBuffer CargarModelo(Context context, String modelName) throws IOException {
        AssetFileDescriptor fileDescriptor = context.getAssets().openFd(modelName);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }
    private List<String> CargarLabel(Context context, String labelsFile) throws IOException {
        List<String> labels = new ArrayList<>();
        BufferedReader reader = new BufferedReader(new InputStreamReader(context.getAssets().open(labelsFile)));
        String line;
        while ((line = reader.readLine()) != null) {
            labels.add(line);
        }
        reader.close();
        return labels;
    }
    public void add_events() {
        btngaleria.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent i = new Intent(Intent.ACTION_PICK,
                        android.provider.MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
                startActivityForResult(i, REQUEST_GALLERY);
            }
        });
    }
    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (resultCode == RESULT_OK && null != data) {
            try {
                Bitmap selectedImage;
                if (requestCode == REQUEST_CAMERA) {
                    selectedImage = (Bitmap) data.getExtras().get("data");
                } else {
                    selectedImage = MediaStore.Images.Media.getBitmap(getContentResolver(), data.getData());
                }
                mSelectedImage = selectedImage;
                mImageView.setImageBitmap(selectedImage);
                ClasificadorDeImagenes(selectedImage);
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }
    }
    private void ClasificadorDeImagenes(Bitmap image) {
        try {
            ByteBuffer buffer = ByteBuffer.allocateDirect(4 * imageSize * imageSize * 3).order(ByteOrder.nativeOrder());
            int[] intValues = new int[imageSize * imageSize];
            image = Bitmap.createScaledBitmap(image, imageSize, imageSize, false);
            image.getPixels(intValues, 0, image.getWidth(), 0, 0, image.getWidth(), image.getHeight());
            for (int pixel : intValues) {
                buffer.putFloat(((pixel >> 16) & 0xFF) * (1.f / 255.f));
                buffer.putFloat(((pixel >> 8) & 0xFF) * (1.f / 255.f));
                buffer.putFloat((pixel & 0xFF) * (1.f / 255.f));
            }
            buffer.rewind();
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 224, 224, 3}, DataType.FLOAT32);
            inputFeature0.loadBuffer(buffer);
            TensorBuffer outputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 4}, DataType.FLOAT32);
            interpreter.run(inputFeature0.getBuffer(), outputFeature0.getBuffer());
            float[] confidences = outputFeature0.getFloatArray();
            int maximaPosicion = 0;
            float maximaparecido = -1;
            for (int i = 0; i < confidences.length; i++) {
                if (confidences[i] > maximaparecido) {
                    maximaparecido = confidences[i];
                    maximaPosicion = i;
                }
            }
            String[] classes = {"Bill Gates", "Elon Musk", "Mark Zuckerberg", "Jeff Bezos"};
            txtParecido.setText(classes[maximaPosicion]);
            StringBuilder resultsText = new StringBuilder("Parecidos:\n");
            for (int i = 0; i < classes.length; i++) {
                resultsText.append(classes[i]).append(": ").append(String.format("%.1f%%", confidences[i] * 100)).append("\n");
            }
            txtResults.setText(resultsText.toString());
        } catch (Exception e) {throw new RuntimeException(e);}
    }
    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        for (int i = 0; i < permissions.length; i++) {
            if (permissions[i].equals(android.Manifest.permission.CAMERA)) {
                btngaleria.setEnabled(grantResults[i] == PackageManager.PERMISSION_GRANTED);
            }
        }
    }
    public ArrayList<String> getPermisosNoAprobados(ArrayList<String> listaPermisos) {
        ArrayList<String> list = new ArrayList<String>();
        Boolean habilitado;
        if (Build.VERSION.SDK_INT >= 23)
            for (String permiso : listaPermisos) {
                if (checkSelfPermission(permiso) != PackageManager.PERMISSION_GRANTED) {
                    list.add(permiso);
                    habilitado = false;
                } else
                    habilitado = true;
                if (permiso.equals(android.Manifest.permission.MANAGE_EXTERNAL_STORAGE) ||
                        permiso.equals(android.Manifest.permission.READ_EXTERNAL_STORAGE))
                    btngaleria.setEnabled(habilitado);
            }
        return list;
    }
}