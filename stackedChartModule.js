var StackedChartModule = function(series, canvas_width, canvas_height) {
    // Create the tag:
    var canvas_tag = "<canvas width='" + canvas_width + "' height='" + canvas_height + "' ";
    canvas_tag += "style='border:1px dotted'></canvas>";
    // Append it to #elements
    var canvas = $(canvas_tag)[0];
    $("#elements").append(canvas);
    // Create the context and the drawing controller:
    var context = canvas.getContext("2d");

    var convertColorOpacity = function(hex) {

        if (hex.indexOf('#') != 0) {
            return 'rgba(0,0,0,0.1)';
        }

        hex = hex.replace('#', '');
        r = parseInt(hex.substring(0, 2), 16);
        g = parseInt(hex.substring(2, 4), 16);
        b = parseInt(hex.substring(4, 6), 16);
        return 'rgba(' + r + ',' + g + ',' + b + ',0.1)';
    };

    // For generating random (distinguishable) colours
    function selectColor(colorNum, colors){
        if (colors < 1) colors = 1; // defaults to one color - avoid divide by zero
        return "hsl(" + (colorNum * (360 / colors) % 360) + ",100%,50%)";
    }

    // Prep the chart properties and series:
    var datasets = []
    for (var i in series) {
        var s = series[i];
        var datasets_number = s.Datasets;

        for (var j=0; j< datasets_number; j++) {
            var color = selectColor(j, datasets_number);
            var pool_nr = j + 1;
            var new_series = {
                label: s.Label + pool_nr,
                borderColor: color,
                backgroundColor: color,
                fill: true,
                data: []
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
        tooltips: {
            mode: 'index',
            intersect: false
        },
        hover: {
            mode: 'nearest',
            intersect: true
        },
        scales: {
            // note: using lists for the axes to accommodate stacking
            xAxis: [{
                stacked: true,
                display: true,
                title: { //todo figure out why axis title doesn't show
                    display: true,
                    text: 'Step'
                },
                ticks: {
                    maxTicksLimit: 11
                }
            }],
            yAxes: [{
                stacked: true,
                display: true,
                title: {
                    display: true,
                    text: 'Size (stake)'
                }
            }]
        }
    };

    var chart = new Chart(context, {
        type: 'line',
        data: chartData,
        options: chartOptions
    });

    this.render = function(data) {
        chart.data.labels.push(control.tick);
        console.log("Data: " + data)
        for (i = 0; i < data.length; i++) {
            chart.data.datasets[i].data.push(data[i]);
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