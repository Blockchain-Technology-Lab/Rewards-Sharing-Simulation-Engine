var StackedChartModule = function(series, canvas_width, canvas_height) {
    // Create the tag:
    var canvas_tag = "<canvas width='" + canvas_width + "' height='" + canvas_height + "' ";
    canvas_tag += "style='border:1px dotted'></canvas>";
    // Append it to #elements
    var canvas = $(canvas_tag)[0];
    $("#elements").append(canvas);
    // Create the context and the drawing controller:
    var context = canvas.getContext("2d");

    var convertColorLightness = function(hslColor) {
        return hslColor.replace(',50%', ',35%');
    };

    function shuffleArray(array) {
        for (let i = array.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [array[i], array[j]] = [array[j], array[i]];
        }
        return array
    }

    // For generating random (distinguishable) colours
    function selectColor(colorNum, colors){
        if (colors < 1) colors = 1; // defaults to one color - avoid divide by zero
        return "hsl(" + ((colorNum * (360 / colors) % 360).toFixed(0)) + ",100%,50%)";
    }

    // Prep the chart properties and series:
    var datasets = []
    for (var i in series) {
        var s = series[i];
        var datasets_number = s.Num_agents;

        for (var color_indexes=[],i=0;i<datasets_number;++i) color_indexes[i]=i;
        color_indexes = shuffleArray(color_indexes)

        for (var j=0; j< datasets_number; j++) {
            var color = selectColor(color_indexes[j], datasets_number);
            var new_series = {
                label: s.Label,
                xLabel: s.xLabel,
                yLabel: s.yLabel,
                tooltipText: s.tooltipText,
                backgroundColor: color,
                borderColor: convertColorLightness(color),
                borderWidth: 0.5,
                fill: true,
                data: [],
                keys: []
            };
            datasets.push(new_series);
        }
    }

    var chartData = {
        labels: [],
        datasets: datasets
    };

    var chartOptions = {
        responsive: true,
        title: {
            display: true,
            text: 'Pool dynamics' //todo make configurable
        },
        interaction: {
          mode: 'nearest',
          axis: 'x',
          intersect: false
        },
        scales: {
            // note: using lists for the axes to accommodate stacking
            xAxes: [{
                stacked: true,
                display: true,
                scaleLabel: {
                    display: true,
                    labelString: datasets[0].xLabel
                },
                ticks: {
                  autoSkip: true,
                  maxTicksLimit: 11
                }
            }],
            yAxes: [{
                stacked: true,
                display: true,
                scaleLabel: {
                    display: true,
                    labelString: datasets[0].yLabel
                },
            }]
        },
        tooltips: {
            displayColors: true,
            position: 'nearest',
            mode: 'index',
            intersect: false,
            // only show active pools (stake > 0) on hover ( maybe do sth similar for legend? )
            filter: function (tooltipItem) {
                return tooltipItem.yLabel !== 0
            },
            callbacks: {
                title: function (tooltipItems, data) {
                    return "Step " + tooltipItems[0].xLabel;
                },
                label: function(tooltipItem, data) {
                    var tooltipText = data.datasets[tooltipItem.datasetIndex].tooltipText;
                    var key = data.datasets[tooltipItem.datasetIndex].keys[tooltipItem.index];
                    return tooltipText + " " + key + ": " + tooltipItem.yLabel.toFixed(4);
                },
                labelColor: function(tooltipItem, chart) {
                    var dataset = chart.config.data.datasets[tooltipItem.datasetIndex];
                    return {
                        backgroundColor : dataset.backgroundColor,
                        borderColor: dataset.borderColor,
                        fill: true
                    }
                }
            }
        },
        elements: {
            point:{
                radius: 0 //to hide points from line
            }
        },
        hover: {
            mode: 'point',
            intersect: true
        },
        legend: {
            display: false
        }

    };

    var chart = new Chart(context, {
        type: 'line',
        data: chartData,
        options: chartOptions
    });

    this.render = function(data) {
        var keys = data[0]
        var values = data[1]
        chart.data.labels.push(control.tick);
        for (i = 0; i < values.length; i++) {
            chart.data.datasets[i].data.push(values[i]);
            chart.data.datasets[i].keys.push(keys[i]);
        }
        chart.update();
    };

    this.reset = function() {
        while (chart.data.labels.length) { chart.data.labels.pop(); }
        chart.data.datasets.forEach(function(dataset) {
            while (dataset.data.length) { dataset.data.pop(); }
        });
        chart.update();
    };
};