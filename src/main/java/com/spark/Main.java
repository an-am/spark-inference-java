package com.spark;
import org.apache.spark.api.java.function.*;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.sql.*;
import org.apache.spark.sql.Encoders;
import org.apache.spark.sql.streaming.*;
import org.apache.spark.sql.types.*;
import org.apache.spark.SparkFiles;
import static org.apache.spark.sql.functions.*;

import java.sql.*;
import java.util.Iterator;

import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import scala.reflect.ClassTag$;


public class Main {
    private static Connection dbConnection = null;
    private static Broadcast<StandardScaler> bScaler = null;
    private static  ThreadLocal<MultiLayerNetwork> modelTL = new ThreadLocal<>();

    // Global constants and configuration
    public static final int NUM_MAX_REQUESTS = 5;
    public static final String HOST = "localhost:9092";
    public static final String TOPIC = "notifications";

    // PostgreSQL configuration
    public static final String DB_URL = "jdbc:postgresql://localhost:5432/postgres";
    public static final String DB_USER = "postgres";
    public static final String DB_PASSWORD = null;

    // Filenames; SparkFiles.get(...) will resolve the real path inside each node's userFiles dir
    public static final String SCALER_PATH = "deep_scaler.npz";
    public static final String MODEL_PATH  = "deep_model.h5";

    public static final double fittedLambdaWealth = 0.1336735055366279;
    public static final double fittedLambdaIncome = 0.3026418664067109;

    // Payload schema for parsing JSON from Kafka
    public static final StructType payloadSchema = new StructType(new StructField[]{
            new StructField("id", DataTypes.IntegerType, true, Metadata.empty()),
            new StructField("client_id", DataTypes.IntegerType, true, Metadata.empty())
    });

    // Full schema for downstream processing (if needed)
    public static final StructType fullSchema = new StructType(new StructField[]{
            new StructField("Id", DataTypes.IntegerType, false, Metadata.empty()),
            new StructField("Age", DataTypes.IntegerType, true, Metadata.empty()),
            new StructField("Gender", DataTypes.IntegerType, true, Metadata.empty()),
            new StructField("FamilyMembers", DataTypes.IntegerType, true, Metadata.empty()),
            new StructField("FinancialEducation", DataTypes.FloatType, true, Metadata.empty()),
            new StructField("RiskPropensity", DataTypes.FloatType, true, Metadata.empty()),
            new StructField("Income", DataTypes.FloatType, true, Metadata.empty()),
            new StructField("Wealth", DataTypes.FloatType, true, Metadata.empty()),
            new StructField("IncomeInvestment", DataTypes.IntegerType, true, Metadata.empty()),
            new StructField("AccumulationInvestment", DataTypes.IntegerType, true, Metadata.empty()),
            new StructField("FinancialStatus", DataTypes.FloatType, true, Metadata.empty()),
            new StructField("ClientId", DataTypes.FloatType, true, Metadata.empty())
    });

    // Get a database connection
    public static Connection getDbConnection() throws Exception {
        if (dbConnection == null || dbConnection.isClosed()) {
            dbConnection = DriverManager.getConnection(DB_URL, DB_USER, DB_PASSWORD);
            dbConnection.setAutoCommit(true);
        }
        return dbConnection;
    }

    // Get the scaler
    public static StandardScaler getScaler() throws Exception {
        return StandardScaler.fromNpyFiles(SparkFiles.get(SCALER_PATH));
    }

    // Get the model
    public static MultiLayerNetwork getModel() throws Exception {
        return KerasModelImport.importKerasSequentialModelAndWeights(SparkFiles.get(MODEL_PATH), false);
    }

    // Data preparation
    public static Dataset<Row> dataPrep(Dataset<Row> df) {
        return df.drop("ClientId")
                .selectExpr("*",
                        "FinancialEducation * log(Wealth) as FinancialStatus",
                        "(POWER(Wealth, " + fittedLambdaWealth + ") - 1) / " + fittedLambdaWealth + " as Wealth",
                        "(POWER(Income, " + fittedLambdaIncome + ") - 1) / " + fittedLambdaIncome + " as Income");
    }

  // Process a micro-batch: perform data preparation, prediction, join, and update DB.
    public static VoidFunction2<Dataset<Row>, Long> processBatch =
        new VoidFunction2<Dataset<Row>, Long>() {

            @Override
            public void call(Dataset<Row> df, Long batch_id) throws Exception {
                Dataset<Row> prepDF = dataPrep(df);

                prepDF
                        .drop("RiskPropensity")
                        .map(
                                (MapFunction<Row, Row>) row -> {
                                    int n = row.size();
                                    INDArray dataIND = Nd4j.create(1, n - 1);

                                    for(int i = 1; i < n; i++) {
                                        double v = ((Number) row.get(i)).doubleValue();
                                        dataIND.putScalar(i-1, v);
                                    }
                                    StandardScaler scaler = bScaler.value();
                                    INDArray scaledIND = scaler.transform(dataIND);

                                    MultiLayerNetwork model = modelTL.get();
                                    if (model == null) {
                                        model = getModel();
                                        modelTL.set(model);
                                    }

                                    double prediction = model.output(scaledIND).getDouble(0);

                                    Object[] newValues = new Object[n + 1];
                                    int predIndex = 5;

                                    for(int i = 0; i < predIndex; i++){
                                        newValues[i] = row.get(i);
                                    }

                                    newValues[predIndex] = prediction;

                                    for(int i = predIndex; i < n; i++){
                                        newValues[i + 1] = row.get(i);
                                    }

                                    return RowFactory.create(newValues);
                                }, Encoders.row(fullSchema))
                        .foreach(
                                (ForeachFunction<Row>) row -> {
                                    queryDb(row);
                                }
                        );
            }
        };


    public static MapGroupsWithStateFunction<Integer, Row, Integer, Row> update_request_count =
            new MapGroupsWithStateFunction<Integer, Row, Integer, Row>() {
                @Override
                public Row call(Integer clientId, Iterator<Row> rows, GroupState<Integer> state) throws Exception {
                    // Process each row in the current batch for this client.
                    while (rows.hasNext()) {
                        // Get the current count from state; default to 0 if no state is present.
                        int currentCount = state.exists() ? state.get() : 0;

                        if (currentCount < NUM_MAX_REQUESTS) {
                            // Increment the state counter.
                            currentCount++;

                            // Update the state with the new count.
                            state.update(currentCount);

                            Row current_row = rows.next();
                            int rowId = current_row.getAs("id");
                            System.out.println("Processing row_id: " + rowId);
                            try {
                                Connection conn = getDbConnection();
                                Statement st = conn.createStatement();

                                String query = "SELECT * FROM needs WHERE id = " + rowId;

                                System.out.println("Executing query: " + query);
                                ResultSet rs = st.executeQuery(query);

                                rs.next();

                                // Retrieve fields from the tuple.
                                Row row =  RowFactory.create(
                                        rs.getInt("id"),
                                        rs.getInt("age"),
                                        rs.getInt("gender"),
                                        rs.getInt("family_members"),
                                        rs.getFloat("financial_education"),
                                        rs.getFloat("risk_propensity"),
                                        rs.getFloat("income"),
                                        rs.getFloat("wealth"),
                                        rs.getFloat("income_investment"),
                                        rs.getInt("accumulation_investment"),
                                        rs.getFloat("financial_status"),
                                        rs.getInt("client_id"));

                                rs.close();
                                st.close();

                                System.out.println(row);
                                return row;

                            } catch (Exception e) {
                                e.printStackTrace();
                            }
                        }
                    }
                    return RowFactory.create((Object) null);
                }
            };

    // Update PostgreSQL for a single row.
    public static void queryDb(Row row) {
        try {
            Connection conn = getDbConnection();
            int id = row.getAs("Id");
            float riskPropensity = row.getAs("RiskPropensity");
            float financialStatus = row.getAs("FinancialStatus");
            String updateQuery = "UPDATE needs SET risk_propensity = ?, financial_status = ? WHERE id = ?";
            PreparedStatement ps = conn.prepareStatement(updateQuery);
            ps.setFloat(1, riskPropensity);
            ps.setFloat(2, financialStatus);
            ps.setInt(3, id);
            ps.executeUpdate();
            System.out.printf("Updated for id = %d risk_propensity = %f and financial_status = %f%n",
                    id, riskPropensity, financialStatus);

            String selectQuery = "SELECT * FROM products WHERE (Income = ? OR Accumulation = ?) AND Risk <= ?";
            PreparedStatement ps2 = conn.prepareStatement(selectQuery);
            int incomeInvestment = row.getAs("IncomeInvestment");
            int accumulationInvestment = row.getAs("AccumulationInvestment");
            ps2.setInt(1, incomeInvestment);
            ps2.setInt(2, accumulationInvestment);
            ps2.setFloat(3, riskPropensity);
            ResultSet rs = ps2.executeQuery();
            int count = 0;
            while (rs.next()) {
                count++;
            }
            System.out.printf("Advised %d products for id = %d%n", count, id);
            ps.close();
            ps2.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }


    public static void main(String[] args) throws Exception {

        SparkSession spark = SparkSession
                .builder()
                .appName("SparkInference")
                .master("spark://Tonys-MacBook-Pro.local:7077")
                .config("spark.jars.excludes", "org.slf4j:slf4j-api")
                .config("spark.driver.bindAddress", "127.0.0.1")
                .getOrCreate();

        spark.sparkContext().setLogLevel("ERROR");

        StandardScaler scaler = getScaler();
        bScaler = spark
                .sparkContext()
                .broadcast(
                        scaler,
                        ClassTag$.MODULE$.apply(StandardScaler.class));

        // Read from Kafka
        Dataset<Row> kafkaDF = spark
                .readStream()
                .format("kafka")
                .option("kafka.bootstrap.servers", HOST)
                .option("subscribe", TOPIC)
                .load();

        System.out.println("Is streaming: " + kafkaDF.isStreaming());

        // Convert binary "value" to string and parse JSON using payloadSchema.
        Dataset<Row> parsedDF = kafkaDF
                .selectExpr("CAST(value AS STRING) as json_value")
                .select(from_json(col("json_value"), payloadSchema).alias("data"))
                .select("data.*")
                .as(Encoders.row(payloadSchema));

        Dataset<Row> statefulDF = parsedDF
                .groupByKey(
                        (MapFunction<Row, Integer>) row -> row.getAs("client_id"),
                        Encoders.INT())
                .mapGroupsWithState(
                        update_request_count,
                        Encoders.INT(),
                        Encoders.row(fullSchema),
                        GroupStateTimeout.NoTimeout()
                        )
                .filter("Id is not null");

        // Process each micro-batch using foreachBatch.
        StreamingQuery query = statefulDF.writeStream()
                .foreachBatch(processBatch)
                .outputMode("update")   // mapGroupsWithState requires Update or Complete
                .option("checkpointLocation", "/tmp/checkpoint")
                .start();

        query.awaitTermination();
    }
}