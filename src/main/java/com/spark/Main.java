package com.spark;
import org.apache.spark.api.java.function.*;
import org.apache.spark.sql.*;
import org.apache.spark.sql.Encoders;
import org.apache.spark.sql.catalyst.expressions.GenericRowWithSchema;
import org.apache.spark.sql.streaming.*;
import org.apache.spark.sql.types.*;
import org.apache.spark.SparkFiles;

import static org.apache.spark.sql.functions.*;



import java.sql.*;
import java.time.Instant;
import java.time.temporal.ChronoField;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.postgresql.ds.PGPoolingDataSource;



public class Main {
    // Executor‑local connection pool that can hand out multiple sockets
    private static PGPoolingDataSource DS = initDataSource();
    private static ThreadLocal<MultiLayerNetwork> modelTL = new ThreadLocal<>();
    private static ThreadLocal<StandardScaler> scalerTL = new ThreadLocal<>();
    private static final AtomicInteger ROWS_DONE  = new AtomicInteger(0);
    private static final AtomicLong START_TIME = new AtomicLong(0);

    // Global constants and configuration
    public static final int NUM_MAX_REQUESTS = 5000;
    public static final String HOST = "localhost:9092";
    public static final String TOPIC = "notifications";

    // PostgreSQL configuration
    public static final String DB_URL = "jdbc:postgresql://localhost:5432/postgres";

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
            new StructField("Id", DataTypes.IntegerType, true, Metadata.empty()),
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
            new StructField("ClientId", DataTypes.IntegerType, true, Metadata.empty())
    });

    public static final StructType predictionSchema = new StructType(new StructField[]{
            new StructField("Id", DataTypes.IntegerType, true, Metadata.empty()),
            new StructField("RiskPropensity", DataTypes.FloatType, true, Metadata.empty()),
            new StructField("IncomeInvestment", DataTypes.IntegerType, true, Metadata.empty()),
            new StructField("AccumulationInvestment", DataTypes.IntegerType, true, Metadata.empty()),
            new StructField("FinancialStatus", DataTypes.FloatType, true, Metadata.empty()),
    });

    public Main() throws SQLException {
    }

    private static PGPoolingDataSource initDataSource() {
        PGPoolingDataSource ds = new PGPoolingDataSource();
        ds.setUrl(DB_URL);
        return ds;
    }

    // Get the scaler
    public static StandardScaler getScaler() throws Exception {
        return StandardScaler.fromNpyFiles(SparkFiles.get(SCALER_PATH));
    }

    // Get the model
    public static MultiLayerNetwork getModel() throws Exception {
        return KerasModelImport.importKerasSequentialModelAndWeights(SparkFiles.get(MODEL_PATH), false);
    }

    public static Dataset<Row> dataPrep(Dataset<Row> df) {

        String incomeExpr = "CAST((POWER(Income,"  + fittedLambdaIncome  + ")-1)/"
                            + fittedLambdaIncome  + " AS float)";
        String wealthExpr = "CAST((POWER(Wealth,"  + fittedLambdaWealth  + ")-1)/"
                            + fittedLambdaWealth  + " AS float)";
        String statusExpr = "CAST(FinancialEducation * log(Wealth) AS float)";

        return df
                .drop("ClientId")                 // remove unused field
                .selectExpr(                      // ORDER must match fullSchema
                        "Id",
                        "Age",
                        "Gender",
                        "FamilyMembers",
                        "FinancialEducation",
                        incomeExpr + " AS Income",
                        wealthExpr + " AS Wealth",
                        "CAST(IncomeInvestment AS INT) AS IncomeInvestment",
                        "CAST(AccumulationInvestment AS INT) AS AccumulationInvestment",
                        statusExpr + " AS FinancialStatus"
                );
    }

  // Process a micro-batch: perform data preparation, prediction, join, and update DB.
    public static VoidFunction2<Dataset<Row>, Long> processBatch =
        new VoidFunction2<Dataset<Row>, Long>() {

            @Override
            public void call(Dataset<Row> df, Long batch_id) throws Exception {
                Dataset<Row> prepDF = dataPrep(df);

                Dataset<Row> predictionDF = prepDF
                        .map(
                                (MapFunction<Row, Row>) row -> {

                                    // !!! Scale in dataprep
                                    StandardScaler scaler = scalerTL.get();

                                    if (scaler == null) {
                                        scaler = getScaler();
                                        scalerTL.set(scaler);
                                    }

                                    MultiLayerNetwork model = modelTL.get();
                                    if (model == null) {
                                        model = getModel();
                                        modelTL.set(model);
                                    }

                                    INDArray dataIND = Nd4j.create(1, 9);

                                    dataIND.putScalar(0, row.getAs("Age"));
                                    dataIND.putScalar(1, row.getAs("Gender"));
                                    dataIND.putScalar(2, row.getAs("FamilyMembers"));
                                    dataIND.putScalar(3, ((Number) row.getAs("FinancialEducation")).floatValue());
                                    dataIND.putScalar(4, ((Number) row.getAs("Income")).floatValue());
                                    dataIND.putScalar(5, ((Number) row.getAs("Wealth")).floatValue());
                                    dataIND.putScalar(6, ((Number) row.getAs("IncomeInvestment")).intValue());
                                    dataIND.putScalar(7, ((Number) row.getAs("AccumulationInvestment")).intValue());
                                    dataIND.putScalar(8, ((Number) row.getAs("FinancialStatus")).floatValue());

                                    INDArray scaledIND = scaler.transform(dataIND);
                                    float prediction = model.output(scaledIND).getFloat(0);

                                    Object[] values = new Object[]{
                                            row.getAs("Id"),
                                            prediction,
                                            ((Number) row.getAs("IncomeInvestment")).intValue(),
                                            ((Number) row.getAs("AccumulationInvestment")).intValue(),
                                            row.getAs("FinancialStatus")
                                    };

                                    return new GenericRowWithSchema(values, predictionSchema);
                                },
                                Encoders.row(predictionSchema));

                predictionDF.foreach(
                        (ForeachFunction<Row>) Main::queryDb
                );

                // how many rows were in this micro-batch?
                long batchRows = df.count();              // df is the original micro-batch
                // Start the stopwatch the first time *any* micro‑batch
                // actually contains data (in case stateful operator saw none)
                if (START_TIME.get() == 0) {
                    START_TIME.compareAndSet(0, System.nanoTime());
                }
                int done = ROWS_DONE.addAndGet((int) batchRows);

                if (done >= NUM_MAX_REQUESTS) {
                    long elapsedNanos = System.nanoTime() - START_TIME.get();
                    double secs = elapsedNanos / 1_000_000_000.0;

                    System.out.printf(
                        "%nProcessed %d rows in %.2f s (%.0f row/s)%n",
                        done, secs, done / secs);

                }
            }
        };


    public static FlatMapGroupsWithStateFunction<Integer, Row, Integer, Row> update_request_count =
            new FlatMapGroupsWithStateFunction<Integer, Row, Integer, Row>() {
                @Override
                public Iterator<Row> call(Integer clientId, Iterator<Row> rows, GroupState<Integer> state) throws Exception {
                    // Process each row in the current batch for this client.
                    // Get the current count from state; default to 0 if no state is present.

                    // first row in the entire job → start the clock
                    if (START_TIME.get() == 0) {
                        START_TIME.compareAndSet(0, System.nanoTime());
                    }

                    int currentCount = state.exists() ? state.get() : 0;
                    List<Integer> ids = new ArrayList<>();

                    while (rows.hasNext()) {
                        Row row = rows.next();
                        int id = row.getAs("id");
                        //System.out.println("currentCount: " + currentCount + " id: " + id);
                        if (currentCount < NUM_MAX_REQUESTS) {
                            currentCount++;
                            ids.add(id);
                        }
                    }

                    if (ids.isEmpty()) {
                        List<Row> rowIterator = new ArrayList<>();
                        Row row = RowFactory.create(null, null, null, null, null, null, null, null, null, null, null, null);
                        rowIterator.add(row);
                        return rowIterator.iterator();
                    }

                    state.update(currentCount);

                    StringBuilder sqlStatement = new StringBuilder("SELECT * FROM needs WHERE id IN (");

                    for (Integer id : ids) {
                        if (id != null) {
                            sqlStatement.append(id).append(",");
                        }
                    }
                    sqlStatement.deleteCharAt(sqlStatement.length() - 1).append(")");

                    Connection conn = DS.getConnection();
                    Statement stmt = conn.createStatement();
                    ResultSet rs = stmt.executeQuery(sqlStatement.toString());

                    List<Row> rowIterator = new ArrayList<>();
                    while (rs.next()) {
                        Row row = RowFactory.create(
                                rs.getInt("id"),
                                rs.getInt("age"),
                                rs.getInt("gender"),
                                rs.getInt("family_members"),
                                rs.getFloat("financial_education"),
                                rs.getFloat("risk_propensity"),
                                rs.getFloat("income"),
                                rs.getFloat("wealth"),
                                rs.getInt("income_investment"),
                                rs.getInt("accumulation_investment"),
                                rs.getFloat("financial_status"),
                                rs.getInt("client_id"));
                        rowIterator.add(row);
                        //System.out.println("row in iterator: " + row);
                    }

                    rs.close();
                    stmt.close();
                    conn.close();

                    return rowIterator.iterator();
                }
            };


    // Update PostgreSQL for a single row.
    public static void queryDb(Row row) {
        try {
            Connection conn = DS.getConnection();

            int id = row.getAs("Id");
            float riskPropensity = row.getAs("RiskPropensity");
            float financialStatus = row.getAs("FinancialStatus");
            int income = row.getAs("IncomeInvestment");
            int accumulation = row.getAs("AccumulationInvestment");

            String updateQuery = "UPDATE needs "
                    + "SET risk_propensity = ?, financial_status = ? "
                    + "WHERE id = ?";

            PreparedStatement ps = conn.prepareStatement(updateQuery);

            ps.setFloat(1, riskPropensity);
            ps.setFloat(2, financialStatus);
            ps.setInt  (3, id);

            int rows = ps.executeUpdate();

            //System.out.printf("id=%d → updated rows=%d%n", id, rows);

            //System.out.printf("Updated for id = %d risk_propensity = %f and financial_status = %f%n",
            //        id, riskPropensity, financialStatus);
            ps.close();

            String selectQuery = "SELECT * "
                    + "FROM products "
                    + "WHERE ("
                      + "Income = ? "
                      + "OR Accumulation = ? ) "
                    + "AND Risk <= ?";

            PreparedStatement ps2 = conn.prepareStatement(selectQuery);
            ps2.setInt  (1, income);
            ps2.setInt  (2, accumulation);
            ps2.setFloat(3, riskPropensity);

            ResultSet rs = ps2.executeQuery();

            int count = 0;
            while (rs.next()) {
                count++;
            }
            //System.out.printf("Advised %d products for id = %d%n", count, id);

            rs.close();
            ps2.close();
            conn.close();

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
                        (MapFunction<Row, Integer>) row -> {
                            // first record anywhere in the job → start stopwatch
                            if (START_TIME.get() == 0) {
                                START_TIME.compareAndSet(0, System.nanoTime());
                            }
                            return row.getAs("client_id");
                        },
                        Encoders.INT())
                .flatMapGroupsWithState(
                        update_request_count,
                        OutputMode.Update(),
                        Encoders.INT(),
                        Encoders.row(fullSchema),
                        GroupStateTimeout.NoTimeout()
                        )
                .filter("Id is not null");

        // Process each micro-batch using foreachBatch.
        StreamingQuery query = statefulDF
                .writeStream()
                .foreachBatch(processBatch)
                .outputMode("update")
                .start();

        query.awaitTermination();
    }
}